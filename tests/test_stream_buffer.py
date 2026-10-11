"""Unit and regression tests for StreamBufferManager and integration contracts."""

import asyncio
from datetime import datetime, timedelta
import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util

from custom_components.llmvision.const import (
    CONF_PROVIDER,
    CONF_STREAM_BUFFER_CAMERAS,
    CONF_STREAM_BUFFER_SECONDS,
    DOMAIN,
)
from custom_components.llmvision.stream_buffer import (
    CameraStreamLease,
    StreamBufferManager,
)


class DummySegment:
    """Mock HLS segment matching Core Segment structure."""

    def __init__(
        self,
        duration: float = 6.0,
        stream_id: int = 1,
        sequence: int = 0,
        start_time: datetime | None = None,
    ):
        self.duration = duration
        self.stream_id = stream_id
        self.sequence = sequence
        self.start_time = start_time or dt_util.utcnow()


def _make_mock_stream(
    preload_stream: bool = False,
    has_outputs: bool = True,
    has_dynamic_settings: bool = True,
    hls_provider: Any = None,
):
    stream = Mock()
    stream.available = True
    stream.start = AsyncMock()
    stream.stop = Mock()

    if has_dynamic_settings:
        stream.dynamic_stream_settings = Mock()
        stream.dynamic_stream_settings.preload_stream = preload_stream
    else:
        stream.dynamic_stream_settings = None

    if hls_provider is None:
        hls_provider = Mock()
        hls_provider.target_duration = 6.0
        hls_provider.idle_timer = Mock()
        hls_provider.idle_timer.start = Mock()
        hls_provider.get_segments = Mock(return_value=[])

    if has_outputs:
        outputs_dict = {"hls": hls_provider}
        stream.outputs = Mock(return_value=outputs_dict)
    else:
        del stream.outputs

    stream.add_provider = Mock(return_value=hls_provider)
    return stream, hls_provider


def _make_entry(
    cameras: list[str] | None = None, buffer_seconds: int = 30, entry_id: str = "entry_1"
):
    entry = Mock()
    entry.entry_id = entry_id
    entry.data = {
        CONF_PROVIDER: "Settings",
        CONF_STREAM_BUFFER_CAMERAS: cameras or ["camera.test"],
        CONF_STREAM_BUFFER_SECONDS: buffer_seconds,
    }
    return entry


@pytest.mark.asyncio
async def test_two_cameras_first_succeeds_second_fails_retains_first():
    """Verify that when second optional camera fails, first camera's lease is retained."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.cam1", "camera.cam2"])

    stream1, hls1 = _make_mock_stream(preload_stream=False)
    stream2, hls2 = _make_mock_stream(preload_stream=False)
    stream2.start = AsyncMock(side_effect=RuntimeError("Cam2 hardware error"))

    cam1 = Mock(async_create_stream=AsyncMock(return_value=stream1))
    cam2 = Mock(async_create_stream=AsyncMock(return_value=stream2))

    def _get_camera(hass_obj, entity_id):
        if entity_id == "camera.cam1":
            return cam1
        if entity_id == "camera.cam2":
            return cam2
        return None

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        side_effect=_get_camera,
    ):
        await manager.async_start()

    assert "camera.cam1" in manager.leases
    assert "camera.cam2" not in manager.leases
    assert stream1.dynamic_stream_settings.preload_stream is True
    assert stream2.dynamic_stream_settings.preload_stream is False
    hls2.idle_timer.start.assert_called_once()


@pytest.mark.asyncio
async def test_two_cameras_second_cancelled_rolls_back_all():
    """Verify that caller cancellation during multi-camera startup releases all prior leases."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.cam1", "camera.cam2"])

    stream1, hls1 = _make_mock_stream(preload_stream=False)
    stream2, hls2 = _make_mock_stream(preload_stream=False)
    gate = asyncio.Event()

    async def _blocking_cam2_start():
        gate.set()
        await asyncio.sleep(10)

    stream2.start = AsyncMock(side_effect=_blocking_cam2_start)

    cam1 = Mock(async_create_stream=AsyncMock(return_value=stream1))
    cam2 = Mock(async_create_stream=AsyncMock(return_value=stream2))

    def _get_camera(hass_obj, entity_id):
        if entity_id == "camera.cam1":
            return cam1
        if entity_id == "camera.cam2":
            return cam2
        return None

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        side_effect=_get_camera,
    ):
        task = asyncio.create_task(manager.async_start())
        await gate.wait()

        # Cam1 was acquired, Cam2 is awaiting
        assert "camera.cam1" in manager.leases

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # Rollback must release Cam1 too
    assert len(manager.leases) == 0
    assert stream1.dynamic_stream_settings.preload_stream is False
    assert stream2.dynamic_stream_settings.preload_stream is False
    hls1.idle_timer.start.assert_called_once()


@pytest.mark.asyncio
async def test_original_preload_true_preserved():
    """Verify that if preload_stream was originally True, releasing leaves it True."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.front"])

    stream, hls = _make_mock_stream(preload_stream=True)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream))

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()

    assert "camera.front" in manager.leases
    assert stream.dynamic_stream_settings.preload_stream is True

    await manager.async_stop()
    assert stream.dynamic_stream_settings.preload_stream is True
    assert hls.idle_timer.start.call_count == 0


@pytest.mark.asyncio
async def test_original_preload_false_restored_and_idle_timer_started():
    """Verify that if preload_stream was False, releasing restores False and starts idle timer."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.front"])

    stream, hls = _make_mock_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream))

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()

    assert "camera.front" in manager.leases
    assert stream.dynamic_stream_settings.preload_stream is True

    await manager.async_stop()
    assert stream.dynamic_stream_settings.preload_stream is False
    hls.idle_timer.start.assert_called_once()


@pytest.mark.asyncio
async def test_async_stop_sets_inactive_before_lock():
    """Verify that async_stop marks is_active=False immediately before acquiring lock."""
    hass = Mock()
    entry = _make_entry(["camera.front"])
    manager = StreamBufferManager(hass, entry)
    manager.is_active = True

    # Lock the manager lock from outside
    await manager._lock.acquire()

    stop_task = asyncio.create_task(manager.async_stop())
    await asyncio.sleep(0.01)

    # is_active should already be False while async_stop is waiting for lock
    assert manager.is_active is False

    # Release the lock and finish stop
    manager._lock.release()
    await stop_task


def test_validation_rejects_missing_or_replaced_hls_provider():
    """Verify validate_lookback rejects when current outputs()["hls"] is missing or replaced."""
    hass = Mock()
    entry = _make_entry(["camera.front"], buffer_seconds=30)

    stream, hls_old = _make_mock_stream(preload_stream=True)
    manager = StreamBufferManager(hass, entry)
    manager.is_active = True
    manager.leases["camera.front"] = CameraStreamLease(
        camera_entity="camera.front",
        stream=stream,
        provider=hls_old,
        original_preload=True,
        borrowed_provider=True,
    )

    # Outputs mapping now returns a different instance
    stream.outputs = Mock(return_value={"hls": Mock()})

    with pytest.raises(ServiceValidationError, match="missing or has been replaced"):
        manager.validate_lookback(["camera.front"], 5)


def test_core_hls_lookback_24s_23s_25s_and_stream_id_mismatch():
    """Verify Core HLS slice logic: 4 completed segments (6s each) + 1 in-progress segment.

    - 24s passes when fresh.
    - 23s passes.
    - 25s fails (duration 24s < 25s).
    - latest segment new stream_id rejects.
    """
    hass = Mock()
    entry = _make_entry(["camera.front"], buffer_seconds=30)

    stream, hls = _make_mock_stream(preload_stream=True)
    hls.target_duration = 6.0
    now = dt_util.utcnow()

    # Deque containing 4 completed segments of 6.0s (total 24.0s) + 1 in-progress segment
    segments = [
        DummySegment(duration=6.0, stream_id=1, sequence=1, start_time=now - timedelta(seconds=25)),
        DummySegment(duration=6.0, stream_id=1, sequence=2, start_time=now - timedelta(seconds=19)),
        DummySegment(duration=6.0, stream_id=1, sequence=3, start_time=now - timedelta(seconds=13)),
        DummySegment(duration=6.0, stream_id=1, sequence=4, start_time=now - timedelta(seconds=7)),
        DummySegment(duration=0.0, stream_id=1, sequence=5, start_time=now - timedelta(seconds=1)),  # current
    ]
    hls.get_segments = Mock(return_value=segments)

    manager = StreamBufferManager(hass, entry)
    manager.is_active = True
    manager.leases["camera.front"] = CameraStreamLease(
        camera_entity="camera.front",
        stream=stream,
        provider=hls,
        original_preload=True,
        borrowed_provider=True,
    )

    # 1. 24s requested: selected_count = min(int(24/6)+1, 5) = 5.
    # Slicing [-6 : -1] yields all 4 completed segments (24.0s) -> PASSES!
    manager.validate_lookback(["camera.front"], 24)

    # 2. 23s requested: selected_count = min(int(23/6)+1, 5) = 4.
    # Slicing [-5 : -1] yields 4 segments (24.0s >= 23s) -> PASSES!
    manager.validate_lookback(["camera.front"], 23)

    # 3. 25s requested: available 24.0s < 25s -> FAILS!
    with pytest.raises(ServiceValidationError, match="Insufficient buffered history"):
        manager.validate_lookback(["camera.front"], 25)

    # 4. Stream restart boundary: current segment has stream_id=2 while history has stream_id=1
    segments_restarted = [
        DummySegment(duration=6.0, stream_id=1, sequence=1, start_time=now - timedelta(seconds=25)),
        DummySegment(duration=6.0, stream_id=1, sequence=2, start_time=now - timedelta(seconds=19)),
        DummySegment(duration=6.0, stream_id=1, sequence=3, start_time=now - timedelta(seconds=13)),
        DummySegment(duration=6.0, stream_id=1, sequence=4, start_time=now - timedelta(seconds=7)),
        DummySegment(duration=0.0, stream_id=2, sequence=1, start_time=now - timedelta(seconds=1)),  # new stream_id!
    ]
    hls.get_segments = Mock(return_value=segments_restarted)

    with pytest.raises(ServiceValidationError, match="crossed a stream restart boundary"):
        manager.validate_lookback(["camera.front"], 24)


def test_validate_lookback_initial_none_recovers_via_core_stream_const():
    """Verify that if module-global MAX_SEGMENTS was None at startup, validate_lookback retries and recovers from Core stream.const."""
    hass = Mock()
    entry = _make_entry(["camera.front"], buffer_seconds=30)

    stream, hls = _make_mock_stream(preload_stream=True)
    hls.target_duration = 6.0
    now = dt_util.utcnow()

    segments = [
        DummySegment(duration=6.0, stream_id=1, sequence=1, start_time=now - timedelta(seconds=25)),
        DummySegment(duration=6.0, stream_id=1, sequence=2, start_time=now - timedelta(seconds=19)),
        DummySegment(duration=6.0, stream_id=1, sequence=3, start_time=now - timedelta(seconds=13)),
        DummySegment(duration=6.0, stream_id=1, sequence=4, start_time=now - timedelta(seconds=7)),
        DummySegment(duration=0.0, stream_id=1, sequence=5, start_time=now - timedelta(seconds=1)),
    ]
    hls.get_segments = Mock(return_value=segments)

    manager = StreamBufferManager(hass, entry)
    manager.is_active = True
    manager.leases["camera.front"] = CameraStreamLease(
        camera_entity="camera.front",
        stream=stream,
        provider=hls,
        original_preload=True,
        borrowed_provider=True,
    )

    # 1. 0s lookback early-returns before attempting capability import
    with patch("custom_components.llmvision.stream_buffer.MAX_SEGMENTS", None):
        manager.validate_lookback(["camera.front"], 0)

    # 2. Positive lookback retries import of real Core stream.const.MAX_SEGMENTS
    with patch("custom_components.llmvision.stream_buffer.MAX_SEGMENTS", None):
        manager.validate_lookback(["camera.front"], 24)


def test_validate_lookback_simulated_import_error_raises_service_validation_error():
    """Verify that when native Core stream MAX_SEGMENTS cannot be imported, positive lookback raises ServiceValidationError without frame processing."""
    hass = Mock()
    entry = _make_entry(["camera.front"], buffer_seconds=30)

    stream, hls = _make_mock_stream(preload_stream=True)
    hls.target_duration = 6.0
    now = dt_util.utcnow()

    segments = [
        DummySegment(duration=6.0, stream_id=1, sequence=1, start_time=now - timedelta(seconds=25)),
        DummySegment(duration=6.0, stream_id=1, sequence=2, start_time=now - timedelta(seconds=19)),
        DummySegment(duration=6.0, stream_id=1, sequence=3, start_time=now - timedelta(seconds=13)),
        DummySegment(duration=6.0, stream_id=1, sequence=4, start_time=now - timedelta(seconds=7)),
        DummySegment(duration=0.0, stream_id=1, sequence=5, start_time=now - timedelta(seconds=1)),
    ]
    hls.get_segments = Mock(return_value=segments)

    manager = StreamBufferManager(hass, entry)
    manager.is_active = True
    manager.leases["camera.front"] = CameraStreamLease(
        camera_entity="camera.front",
        stream=stream,
        provider=hls,
        original_preload=True,
        borrowed_provider=True,
    )

    import builtins
    orig_import = builtins.__import__

    def _import_intercept(name, *args, **kwargs):
        if name == "homeassistant.components.stream.const":
            raise ImportError("Simulated missing stream.const capability")
        return orig_import(name, *args, **kwargs)

    with patch("custom_components.llmvision.stream_buffer.MAX_SEGMENTS", None):
        with patch("builtins.__import__", side_effect=_import_intercept):
            # 0s lookback still early returns without error
            manager.validate_lookback(["camera.front"], 0)

            # Positive lookback fails capability guard explicitly
            with pytest.raises(
                ServiceValidationError,
                match="Home Assistant stream component MAX_SEGMENTS capability is unavailable.",
            ):
                manager.validate_lookback(["camera.front"], 24)


def test_services_yaml_max_tokens_and_lookback_selectors():
    """Regression test ensuring max_tokens selector is number in all services and lookback is only in stream_analyzer."""
    import yaml

    with open("custom_components/llmvision/services.yaml", encoding="utf-8") as f:
        services = yaml.safe_load(f)

    # Video analyzer has valid number selector for max_tokens and NO lookback field
    video_fields = services["video_analyzer"]["fields"]
    assert "number" in video_fields["max_tokens"]["selector"]
    assert "lookback" not in video_fields

    # Stream analyzer has valid selectors
    stream_fields = services["stream_analyzer"]["fields"]
    assert "number" in stream_fields["max_tokens"]["selector"]
    assert "number" in stream_fields["lookback"]["selector"]
    assert "0-60s" in stream_fields["lookback"]["description"]


# ---------------------------------------------------------------------------
# Focused regression tests for same-stream provider recovery, periodic detection,
# healthy no-churn, and lifecycle / error isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def deterministic_timer():
    """Deterministic timer fixture patching async_track_time_interval to avoid real timer interactions."""
    unsub = Mock()
    with patch(
        "custom_components.llmvision.stream_buffer.async_track_time_interval",
        return_value=unsub,
    ) as mock_track:
        yield mock_track


def _make_realistic_stream(preload_stream: bool = False, initial_hls: Any = None):
    """Create a realistic mock stream whose outputs() reflects add_provider and dynamic mutations."""
    stream = Mock()
    stream.available = True
    stream.start = AsyncMock()
    stream.stop = Mock()
    stream.dynamic_stream_settings = Mock()
    stream.dynamic_stream_settings.preload_stream = preload_stream

    outputs_dict = {}
    if initial_hls is None:
        initial_hls = Mock()
        initial_hls.target_duration = 6.0
        initial_hls.idle_timer = Mock()
        initial_hls.idle_timer.start = Mock()
        initial_hls.get_segments = Mock(return_value=[])

    outputs_dict["hls"] = initial_hls
    stream.outputs = Mock(side_effect=lambda: dict(outputs_dict))

    def _add_provider(name):
        new_provider = Mock()
        new_provider.target_duration = 6.0
        new_provider.idle_timer = Mock()
        new_provider.idle_timer.start = Mock()
        new_provider.get_segments = Mock(return_value=[])
        outputs_dict[name] = new_provider
        return new_provider

    stream.add_provider = Mock(side_effect=_add_provider)
    stream._outputs_dict = outputs_dict
    return stream, initial_hls


@pytest.mark.asyncio
async def test_timer_registration_callback_and_stop_unsubscribe(deterministic_timer):
    """Verify actual timer registration, periodic callback execution, and unsubscribe on stop."""
    hass = Mock()
    hass.is_running = True
    hass.async_create_task = Mock(side_effect=lambda coro: asyncio.create_task(coro))
    hass.states.get = Mock(return_value=Mock(state="idle"))
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()

        # 1. Verify timer registration with 30s interval
        deterministic_timer.assert_called_once()
        args, _ = deterministic_timer.call_args
        assert args[0] is hass
        timer_callback = args[1]
        assert args[2] == timedelta(seconds=30)
        unsub_handle = deterministic_timer.return_value
        assert manager._unsub_reconcile is unsub_handle

        # 2. Invoke periodic timer callback
        timer_callback()
        assert manager._reconcile_task is not None
        await manager._reconcile_task

        # No churn on healthy lease
        assert manager.leases["camera.test_1"].provider is hls_provider

        # 3. Stop manager and verify unsubscribe
        await manager.async_stop()
        unsub_handle.assert_called_once()
        assert manager._unsub_reconcile is None
        assert manager.is_active is False


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["missing", "replacement"])
async def test_periodic_reconciliation_recovers_missing_and_replacement_provider(scenario):
    """Verify periodic reconciliation recovers when HLS provider is missing or replaced on the same stream."""
    hass = Mock()
    hass.is_running = True
    hass.states.get = Mock(return_value=Mock(state="idle"))
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider1 = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()
        assert manager.leases["camera.test_1"].provider is hls_provider1
        assert manager.leases["camera.test_1"].provider is stream.outputs()["hls"]

        if scenario == "missing":
            # Core worker stopped and stripped outputs
            stream._outputs_dict.clear()
            assert stream.outputs() == {}
        elif scenario == "replacement":
            # Core allocated a new provider instance on the same stream
            new_provider = Mock()
            new_provider.target_duration = 6.0
            new_provider.idle_timer = Mock()
            new_provider.idle_timer.start = Mock()
            new_provider.get_segments = Mock(return_value=[])
            stream._outputs_dict["hls"] = new_provider
            assert stream.outputs()["hls"] is new_provider

        # Run periodic reconciliation
        await manager._async_reconcile_all()

        # Manager must detect change, update lease, and lease.provider must be stream.outputs()['hls']
        assert "camera.test_1" in manager.leases
        lease = manager.leases["camera.test_1"]
        assert lease.provider is stream.outputs()["hls"]
        assert lease.provider is not hls_provider1
        assert lease.stream is stream


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["missing", "replacement"])
async def test_state_change_recovers_missing_and_replacement_provider(scenario):
    """Verify state change recovers lease when current HLS output is missing or replaced on the same stream."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider1 = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()
        assert manager.leases["camera.test_1"].provider is hls_provider1

        if scenario == "missing":
            stream._outputs_dict.clear()
            assert stream.outputs() == {}
        elif scenario == "replacement":
            new_provider = Mock()
            new_provider.target_duration = 6.0
            new_provider.idle_timer = Mock()
            new_provider.idle_timer.start = Mock()
            new_provider.get_segments = Mock(return_value=[])
            stream._outputs_dict["hls"] = new_provider
            assert stream.outputs()["hls"] is new_provider

        event = Mock()
        event.data = {
            "entity_id": "camera.test_1",
            "new_state": Mock(state="idle"),
        }
        await manager._async_handle_camera_state_change(event)

        assert "camera.test_1" in manager.leases
        lease = manager.leases["camera.test_1"]
        assert lease.provider is stream.outputs()["hls"]
        assert lease.provider is not hls_provider1
        assert lease.stream is stream


@pytest.mark.asyncio
@pytest.mark.parametrize("state_val", ["unavailable", "unknown", None])
async def test_offline_camera_skips_acquisition_and_releases_stale_lease(state_val):
    """Verify reconciliation consults hass.states and releases/skips offline or missing cameras."""
    hass = Mock()
    hass.is_running = True
    state_mock = Mock(state=state_val) if state_val is not None else None
    hass.states.get = Mock(return_value=state_mock)
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        # 1. Healthy start with state="idle"
        hass.states.get.return_value = Mock(state="idle")
        await manager.async_start()
        assert "camera.test_1" in manager.leases

        # 2. Camera goes offline / unknown / missing
        hass.states.get.return_value = state_mock
        camera.async_create_stream.reset_mock()

        # Reconcile runs
        await manager._async_reconcile_all()

        # Lease is cleanly released and NOT re-acquired
        assert "camera.test_1" not in manager.leases
        camera.async_create_stream.assert_not_called()
        assert stream.dynamic_stream_settings.preload_stream is False
        hls_provider.idle_timer.start.assert_called_once()


@pytest.mark.asyncio
async def test_healthy_lease_no_churn_on_state_change_and_reconcile():
    """Verify that a healthy lease is retained without churn by state events or periodic reconciliation."""
    hass = Mock()
    hass.is_running = True
    hass.states.get = Mock(return_value=Mock(state="idle"))
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()
        assert "camera.test_1" in manager.leases
        lease_before = manager.leases["camera.test_1"]
        assert camera.async_create_stream.call_count == 1

        # 1. State change event with healthy new_state
        event = Mock()
        event.data = {
            "entity_id": "camera.test_1",
            "new_state": Mock(state="idle"),
        }
        await manager._async_handle_camera_state_change(event)

        assert manager.leases["camera.test_1"] is lease_before
        assert camera.async_create_stream.call_count == 1
        assert hls_provider.idle_timer.start.call_count == 0

        # 2. Periodic reconciliation
        await manager._async_reconcile_all()

        assert manager.leases["camera.test_1"] is lease_before
        assert camera.async_create_stream.call_count == 1
        assert hls_provider.idle_timer.start.call_count == 0


@pytest.mark.asyncio
async def test_in_flight_stop_cancellation_rollback_and_no_post_stop():
    """Verify in-flight stop cancels reconcile task, triggers rollback, and prevents post-stop acquisition."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    gate = asyncio.Event()

    async def _blocking_stream_start():
        gate.set()
        await asyncio.sleep(10)

    stream.start = AsyncMock(side_effect=_blocking_stream_start)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        manager.is_active = True
        reconcile_task = asyncio.create_task(
            manager._async_reconcile_camera("camera.test_1")
        )
        manager._reconcile_task = reconcile_task

        await gate.wait()
        # Stream preload was set before start
        assert stream.dynamic_stream_settings.preload_stream is True

        # Stop manager while reconciliation is blocked in stream.start()
        await manager.async_stop()

        # Task was cancelled, rollback occurred
        assert manager.is_active is False
        assert len(manager.leases) == 0
        assert stream.dynamic_stream_settings.preload_stream is False
        hls_provider.idle_timer.start.assert_called_once()

        # Post-stop queued tick or callback must not acquire
        manager._on_reconcile_interval()
        assert manager._reconcile_task is None
        await manager._async_reconcile_camera("camera.test_1")
        assert len(manager.leases) == 0


@pytest.mark.asyncio
async def test_shared_stream_preload_preserved_during_recovery_and_stop():
    """Verify that if stream originally had preload_stream=True, recovery and stop preserve it."""
    hass = Mock()
    hass.is_running = True
    hass.states.get = Mock(return_value=Mock(state="idle"))
    entry = _make_entry(["camera.test_shared"])

    stream, hls_provider1 = _make_realistic_stream(preload_stream=True)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()
        assert manager.leases["camera.test_shared"].original_preload is True
        assert stream.dynamic_stream_settings.preload_stream is True

        # Invalidate outputs to trigger recovery
        stream._outputs_dict.clear()
        await manager._async_reconcile_all()

        # Preload_stream must remain True, idle_timer never started
        assert stream.dynamic_stream_settings.preload_stream is True
        assert hls_provider1.idle_timer.start.call_count == 0
        assert manager.leases["camera.test_shared"].original_preload is True

        # Stop manager
        await manager.async_stop()

        # Still preserved! Must NOT stop or arm idle timer for shared stream
        assert stream.dynamic_stream_settings.preload_stream is True
        assert hls_provider1.idle_timer.start.call_count == 0


@pytest.mark.asyncio
async def test_async_start_race_with_async_stop_no_listener_leak():
    """Verify that when async_stop is called while async_start awaits stream.start, listeners do not leak."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    start_barrier = asyncio.Event()
    release_barrier = asyncio.Event()

    async def _blocking_stream_start():
        start_barrier.set()
        await release_barrier.wait()

    stream.start = AsyncMock(side_effect=_blocking_stream_start)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ), patch(
        "custom_components.llmvision.stream_buffer.async_track_state_change_event"
    ) as mock_track_state, patch(
        "custom_components.llmvision.stream_buffer.async_track_time_interval"
    ) as mock_track_interval:
        # Start async_start in background
        start_task = asyncio.create_task(manager.async_start())

        # Wait until async_start reaches stream.start
        await asyncio.wait_for(start_barrier.wait(), timeout=2.0)

        # Launch stop while async_start is holding lock
        stop_task = asyncio.create_task(manager.async_stop())

        # Yield to allow async_stop to run up to lock acquisition (setting is_active = False)
        await asyncio.sleep(0.01)
        assert manager.is_active is False

        # Release stream.start so async_start finishes acquisition loop
        release_barrier.set()

        # Await both tasks bounded
        await asyncio.wait_for(asyncio.gather(start_task, stop_task), timeout=2.0)

        # Assert clean state
        assert manager.is_active is False
        assert len(manager.leases) == 0
        assert manager._unsub_reconcile is None
        assert manager._unsub_state_change is None
        assert manager._unsub_ha_started is None
        assert manager._reconcile_task is None

        # No listeners leaked during startup race
        mock_track_state.assert_not_called()
        mock_track_interval.assert_not_called()
        assert stream.dynamic_stream_settings.preload_stream is False


@pytest.mark.asyncio
async def test_reconcile_interval_no_overlapping_tasks_and_stop_no_reacquire():
    """Verify that multiple interval ticks do not create overlapping tasks and post-stop does not reacquire."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.test_1"])

    stream, hls_provider = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    gate = asyncio.Event()

    async def _blocking_reconcile():
        gate.set()
        await asyncio.sleep(10)

    tasks_created = []

    def _async_create_task(coro):
        t = asyncio.create_task(coro)
        tasks_created.append(t)
        return t

    hass.async_create_task = Mock(side_effect=_async_create_task)
    hass.states.get = Mock(return_value=Mock(state="idle"))

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()
        hass.async_create_task.reset_mock()
        tasks_created.clear()

        # Patch _async_reconcile_all to block
        with patch.object(
            manager, "_async_reconcile_all", side_effect=_blocking_reconcile
        ):
            # First tick creates a task
            manager._on_reconcile_interval()
            assert hass.async_create_task.call_count == 1
            await gate.wait()

            # Second tick while first is still running must NOT create another task
            manager._on_reconcile_interval()
            assert hass.async_create_task.call_count == 1

        # Stop manager cancels running task and clears reference
        await manager.async_stop()
        assert manager.is_active is False
        assert manager._reconcile_task is None
        assert len(manager.leases) == 0

        # Post-stop tick does nothing
        manager._on_reconcile_interval()
        assert hass.async_create_task.call_count == 1
        assert manager._reconcile_task is None


@pytest.mark.asyncio
async def test_strict_validation_after_missing_recovery_empty_segments_fails():
    """Verify that after recovering from missing provider, validate_lookback strictly rejects if segments are empty."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.test_1"], buffer_seconds=30)

    stream, hls_provider1 = _make_realistic_stream(preload_stream=False)
    camera = Mock(async_create_stream=AsyncMock(return_value=stream), stream=stream)

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        return_value=camera,
    ):
        await manager.async_start()

        # Invalidate outputs (missing provider)
        stream._outputs_dict.clear()

        # Reconcile acquires a new provider via add_provider, but that new provider has empty segments
        await manager._async_reconcile_all()

        assert "camera.test_1" in manager.leases
        lease = manager.leases["camera.test_1"]
        assert lease.provider is stream.outputs()["hls"]
        assert lease.provider is not hls_provider1

        # 0s lookback succeeds
        manager.validate_lookback(["camera.test_1"], 0)

        # Positive lookback must strictly raise ServiceValidationError due to empty segments (stream warming up)
        with pytest.raises(ServiceValidationError, match="warming up"):
            manager.validate_lookback(["camera.test_1"], 5)


@pytest.mark.asyncio
async def test_error_isolation_failed_camera_first_healthy_camera_second():
    """Verify that failure on first camera does not abort reconciliation of second camera, and Cancellation is not swallowed."""
    hass = Mock()
    hass.is_running = True
    entry = _make_entry(["camera.cam1_failing", "camera.cam2_healthy"])

    stream1, hls1 = _make_realistic_stream(preload_stream=False)
    stream2, hls2 = _make_realistic_stream(preload_stream=False)

    cam1 = Mock(
        async_create_stream=AsyncMock(side_effect=RuntimeError("Cam1 hardware down")),
        stream=stream1,
    )
    cam2 = Mock(async_create_stream=AsyncMock(return_value=stream2), stream=stream2)

    def _get_camera(hass_obj, entity_id):
        if entity_id == "camera.cam1_failing":
            return cam1
        if entity_id == "camera.cam2_healthy":
            return cam2
        return None

    manager = StreamBufferManager(hass, entry)
    with patch.object(
        importlib.import_module("homeassistant.components.camera"),
        "get_camera_from_entity_id",
        side_effect=_get_camera,
    ):
        # Startup attempts both: cam1 fails, cam2 succeeds
        await manager.async_start()

        assert "camera.cam1_failing" not in manager.leases
        assert "camera.cam2_healthy" in manager.leases
        assert (
            manager.leases["camera.cam2_healthy"].provider
            is stream2.outputs()["hls"]
        )

        # Run reconciliation: cam1 still fails, cam2 must still be checked and remain healthy
        await manager._async_reconcile_all()

        assert "camera.cam1_failing" not in manager.leases
        assert "camera.cam2_healthy" in manager.leases
        assert (
            manager.leases["camera.cam2_healthy"].provider
            is stream2.outputs()["hls"]
        )

        # Verify CancelledError in cam1 is NOT swallowed, but propagates
        with patch.object(
            cam1, "async_create_stream", side_effect=asyncio.CancelledError()
        ):
            with pytest.raises(asyncio.CancelledError):
                await manager._async_reconcile_all()

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

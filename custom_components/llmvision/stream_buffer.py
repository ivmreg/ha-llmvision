"""Stream buffer keeper and lookback manager for LLM Vision."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import callback
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.util import dt as dt_util

from .const import (
    CONF_STREAM_BUFFER_CAMERAS,
    CONF_STREAM_BUFFER_SECONDS,
    DEFAULT_STREAM_BUFFER_SECONDS,
)

_LOGGER = logging.getLogger(__name__)

# Native Core Stream MAX_SEGMENTS import
try:
    from homeassistant.components.stream.const import MAX_SEGMENTS
except ImportError:
    MAX_SEGMENTS = None

DEFAULT_RECONCILE_INTERVAL_SECONDS = 30


@dataclass
class CameraStreamLease:
    """Represents an active stream buffer lease for a camera entity."""

    camera_entity: str
    stream: Any
    provider: Any
    original_preload: bool
    borrowed_provider: bool


class StreamBufferManager:
    """Manages continuous HLS pre-buffering and lookback validation for configured cameras."""

    def __init__(self, hass, entry) -> None:
        """Initialize the stream buffer manager."""
        self.hass = hass
        self.entry = entry
        self.entry_id = entry.entry_id
        self.leases: dict[str, CameraStreamLease] = {}
        self.is_active: bool = False
        self._lock = asyncio.Lock()
        self._unsub_state_change = None
        self._unsub_ha_started = None
        self._unsub_reconcile = None
        self._reconcile_task: asyncio.Task | None = None

    @property
    def buffered_cameras(self) -> list[str]:
        """Return list of camera entity IDs configured for buffering."""
        cameras = self.entry.data.get(CONF_STREAM_BUFFER_CAMERAS) or []
        if isinstance(cameras, str):
            return [cameras]
        return list(cameras)

    @property
    def max_buffer_seconds(self) -> int:
        """Return configured maximum requested lookback in seconds."""
        return int(
            self.entry.data.get(
                CONF_STREAM_BUFFER_SECONDS, DEFAULT_STREAM_BUFFER_SECONDS
            )
        )

    async def async_start(self) -> None:
        """Start pre-buffering for configured cameras."""
        async with self._lock:
            if self.is_active:
                return
            self.is_active = True

            try:
                # Acquire streams for all configured cameras
                for camera_entity in self.buffered_cameras:
                    await self._async_acquire_stream(camera_entity)

                if not self.is_active:
                    return

                # Retry after Home Assistant startup if not currently running
                if not getattr(self.hass, "is_running", True):
                    self._unsub_ha_started = self.hass.bus.async_listen_once(
                        EVENT_HOMEASSISTANT_STARTED, self._async_on_ha_started
                    )

                # Listen for camera state changes (availability, replacement)
                if self.buffered_cameras:
                    try:
                        self._unsub_state_change = async_track_state_change_event(
                            self.hass,
                            self.buffered_cameras,
                            self._async_handle_camera_state_change,
                        )
                    except Exception as e:
                        _LOGGER.debug(
                            "LLM Vision: Could not register camera state listener: %s",
                            e,
                        )

                    try:
                        self._unsub_reconcile = async_track_time_interval(
                            self.hass,
                            self._on_reconcile_interval,
                            timedelta(seconds=DEFAULT_RECONCILE_INTERVAL_SECONDS),
                        )
                    except Exception as e:
                        _LOGGER.debug(
                            "LLM Vision: Could not register periodic stream reconciler: %s",
                            e,
                        )
            except BaseException:
                # Rollback all prior leases and cleanup state on cancellation or fatal error during startup
                self.is_active = False
                if self._unsub_reconcile:
                    try:
                        self._unsub_reconcile()
                    except Exception:
                        pass
                    self._unsub_reconcile = None
                if self._unsub_state_change:
                    try:
                        self._unsub_state_change()
                    except Exception:
                        pass
                    self._unsub_state_change = None
                if self._unsub_ha_started:
                    try:
                        self._unsub_ha_started()
                    except Exception:
                        pass
                    self._unsub_ha_started = None
                if self._reconcile_task is not None and not self._reconcile_task.done():
                    self._reconcile_task.cancel()
                    try:
                        await self._reconcile_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self._reconcile_task = None
                for camera_entity in list(self.leases.keys()):
                    await self.async_release_camera(camera_entity)
                self.leases.clear()
                raise

    @callback
    def _on_reconcile_interval(self, now: Any = None) -> None:
        """Periodic callback to schedule background stream reconciliation."""
        if not self.is_active:
            return
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return
        self._reconcile_task = self.hass.async_create_task(self._async_reconcile_all())

    async def _async_reconcile_all(self) -> None:
        """Reconcile stream leases for all configured cameras to detect invalidations without state events."""
        for camera_entity in self.buffered_cameras:
            if not self.is_active:
                break
            try:
                await self._async_reconcile_camera(camera_entity)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                _LOGGER.debug(
                    "LLM Vision: Error reconciling camera %s: %s",
                    camera_entity,
                    err,
                )

    async def _async_reconcile_camera(
        self, camera_entity: str, new_state: Any = None
    ) -> None:
        """Reconcile a single camera stream lease against current state and outputs."""
        async with self._lock:
            if not self.is_active:
                return

            # Consult entity state (from state event or hass.states)
            state_obj = (
                new_state
                if new_state is not None
                else self.hass.states.get(camera_entity)
            )
            state_val = getattr(state_obj, "state", None) if state_obj is not None else None

            if state_obj is None or state_val in ("unavailable", "unknown"):
                # Camera is offline, unknown, or missing; release lease cleanly and do not acquire
                if camera_entity in self.leases:
                    await self.async_release_camera(camera_entity)
                return

            existing_lease = self.leases.get(camera_entity)
            if existing_lease is not None:
                try:
                    from homeassistant.components.camera import (
                        get_camera_from_entity_id,
                    )

                    camera = get_camera_from_entity_id(self.hass, camera_entity)
                    current_stream = getattr(camera, "stream", None)
                    if (
                        current_stream is not None
                        and current_stream is existing_lease.stream
                    ):
                        outputs = (
                            current_stream.outputs()
                            if callable(current_stream.outputs)
                            else getattr(current_stream, "outputs", {})
                        )
                        current_hls = (
                            outputs.get("hls")
                            if (
                                isinstance(outputs, (dict, Mapping))
                                or hasattr(outputs, "get")
                            )
                            else None
                        )
                        if (
                            current_hls is not None
                            and current_hls is existing_lease.provider
                        ):
                            # Same stream instance and same active HLS provider; healthy lease, retain without churn
                            return
                except Exception:
                    pass

                # Stream was replaced, camera reloaded, or HLS provider missing/replaced; release old lease cleanly
                await self.async_release_camera(camera_entity)

            await self._async_acquire_stream(camera_entity)

    async def _async_on_ha_started(self, event=None) -> None:
        """Handle Home Assistant started event to retry stream acquisition."""
        async with self._lock:
            self._unsub_ha_started = None
            if not self.is_active:
                return
            for camera_entity in self.buffered_cameras:
                if camera_entity not in self.leases:
                    await self._async_acquire_stream(camera_entity)

    async def _async_handle_camera_state_change(self, event) -> None:
        """Handle state change events for configured cameras."""
        if not self.is_active:
            return

        event_data = getattr(event, "data", {})
        entity_id = event_data.get("entity_id")
        if not entity_id or entity_id not in self.buffered_cameras:
            return

        new_state = event_data.get("new_state")
        await self._async_reconcile_camera(entity_id, new_state=new_state)

    async def _async_acquire_stream(self, camera_entity: str) -> None:
        """Acquire or borrow an HLS stream provider for a camera entity with active guards."""
        if not self.is_active:
            return

        try:
            from homeassistant.components.camera import get_camera_from_entity_id
        except ImportError as err:
            _LOGGER.error(
                "LLM Vision: Failed to import Home Assistant camera component: %s",
                err,
            )
            return

        try:
            camera = get_camera_from_entity_id(self.hass, camera_entity)
        except Exception as err:
            _LOGGER.debug(
                "LLM Vision: Error looking up camera entity %s: %s",
                camera_entity,
                err,
            )
            return

        if camera is None:
            _LOGGER.debug(
                "LLM Vision: Camera entity %s not found in registry; will retry on state change",
                camera_entity,
            )
            return

        try:
            stream = await camera.async_create_stream()
        except Exception as err:
            _LOGGER.warning(
                "LLM Vision: Could not create stream for optional camera %s: %s",
                camera_entity,
                err,
            )
            return

        if not self.is_active:
            return

        if not stream:
            _LOGGER.debug(
                "LLM Vision: Camera %s does not provide a stream", camera_entity
            )
            return

        # Validate required native Core capabilities
        if (
            not hasattr(stream, "dynamic_stream_settings")
            or stream.dynamic_stream_settings is None
            or not hasattr(stream.dynamic_stream_settings, "preload_stream")
        ):
            _LOGGER.warning(
                "LLM Vision: Camera %s stream does not support dynamic stream settings (preload_stream); lookback buffering unsupported.",
                camera_entity,
            )
            return

        if not hasattr(stream, "outputs"):
            _LOGGER.warning(
                "LLM Vision: Camera %s stream does not expose public outputs mapping; lookback buffering unsupported.",
                camera_entity,
            )
            return

        # Record original preload flag
        orig_preload = bool(
            getattr(stream.dynamic_stream_settings, "preload_stream", False)
        )
        stream.dynamic_stream_settings.preload_stream = True

        provider = None
        borrowed = False

        def _rollback():
            if not orig_preload:
                stream.dynamic_stream_settings.preload_stream = False
                if (
                    provider is not None
                    and hasattr(provider, "idle_timer")
                    and hasattr(provider.idle_timer, "start")
                ):
                    try:
                        provider.idle_timer.start()
                    except Exception as timer_err:
                        _LOGGER.debug(
                            "Error arming idle timer on rollback: %s", timer_err
                        )

        try:
            outputs = (
                stream.outputs()
                if callable(stream.outputs)
                else getattr(stream, "outputs", {})
            )
            if isinstance(outputs, (dict, Mapping)) or hasattr(outputs, "get"):
                provider = outputs.get("hls")

            if provider is not None:
                borrowed = True
            else:
                provider = stream.add_provider("hls")

            await stream.start()

            # Active check after await
            if not self.is_active:
                _rollback()
                return

            self.leases[camera_entity] = CameraStreamLease(
                camera_entity=camera_entity,
                stream=stream,
                provider=provider,
                original_preload=orig_preload,
                borrowed_provider=borrowed,
            )
            _LOGGER.debug(
                "LLM Vision: Started stream buffer keeper for %s", camera_entity
            )

        except asyncio.CancelledError:
            _rollback()
            raise
        except Exception as err:
            _rollback()
            _LOGGER.warning(
                "LLM Vision: Failed to acquire stream lease for optional camera %s: %s",
                camera_entity,
                err,
            )
            return

    async def async_release_camera(self, camera_entity: str) -> None:
        """Release stream lease for a single camera, restoring original settings."""
        lease = self.leases.pop(camera_entity, None)
        if lease is None:
            return

        stream = lease.stream
        # Restore original preload flag
        if (
            hasattr(stream, "dynamic_stream_settings")
            and stream.dynamic_stream_settings is not None
        ):
            if not lease.original_preload:
                stream.dynamic_stream_settings.preload_stream = False

        # Allow native idle expiry when preload was originally False
        if not lease.original_preload and lease.provider is not None:
            idle_timer = getattr(lease.provider, "idle_timer", None)
            if idle_timer is not None and hasattr(idle_timer, "start"):
                try:
                    idle_timer.start()
                except Exception as err:
                    _LOGGER.debug(
                        "LLM Vision: Error arming idle timer for %s: %s",
                        camera_entity,
                        err,
                    )

        _LOGGER.debug("LLM Vision: Released stream lease for %s", camera_entity)

    async def async_stop(self) -> None:
        """Stop all stream keepers and unsubscribe listeners."""
        # Set is_active False BEFORE waiting for lock so queued startup/callbacks abort early
        self.is_active = False

        # Cancel and await owned reconciliation task BEFORE waiting for lock
        if self._reconcile_task is not None and not self._reconcile_task.done():
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reconcile_task = None

        async with self._lock:
            if self._unsub_reconcile:
                try:
                    self._unsub_reconcile()
                except Exception:
                    pass
                self._unsub_reconcile = None

            if self._unsub_state_change:
                try:
                    self._unsub_state_change()
                except Exception:
                    pass
                self._unsub_state_change = None

            if self._unsub_ha_started:
                try:
                    self._unsub_ha_started()
                except Exception:
                    pass
                self._unsub_ha_started = None

            for camera_entity in list(self.leases.keys()):
                await self.async_release_camera(camera_entity)
            self.leases.clear()

    def validate_lookback(
        self, image_entities: list[str] | str, lookback: Any
    ) -> None:
        """Strictly validate requested lookback, freshness, and buffer readiness against Core HLS constraints."""
        # 1. Type and bounds validation
        if isinstance(lookback, bool):
            raise ServiceValidationError(
                f"Invalid lookback value '{lookback}': boolean values are not permitted."
            )

        if isinstance(lookback, float) and not lookback.is_integer():
            raise ServiceValidationError(
                f"Invalid lookback value '{lookback}': lookback must be an integer between 0 and 60 seconds."
            )

        try:
            lookback_val = int(lookback)
        except (ValueError, TypeError, OverflowError):
            raise ServiceValidationError(
                f"Invalid lookback value '{lookback}': must be a finite integer between 0 and 60 seconds."
            )

        if lookback_val < 0 or lookback_val > 60:
            raise ServiceValidationError(
                f"Requested lookback ({lookback_val}s) must be between 0 and 60 seconds."
            )

        # 0s lookback means no lookback requested
        if lookback_val == 0:
            return

        # 2. Check Core MAX_SEGMENTS capability
        max_segments = MAX_SEGMENTS
        if max_segments is None:
            try:
                from homeassistant.components.stream.const import (
                    MAX_SEGMENTS as max_segments,
                )
            except ImportError as err:
                raise ServiceValidationError(
                    "Home Assistant stream component MAX_SEGMENTS capability is unavailable."
                ) from err

        # 3. Runtime manager activity check
        if not self.is_active:
            raise ServiceValidationError(
                "Stream buffer manager is not active. Pre-event lookback requires an active stream buffer."
            )

        # 4. Entity check
        if isinstance(image_entities, str):
            entities = [image_entities]
        elif isinstance(image_entities, (list, tuple)):
            entities = list(image_entities)
        else:
            entities = []

        if not entities:
            raise ServiceValidationError(
                "No camera entity provided for stream lookback analysis."
            )

        for camera_entity in entities:
            if camera_entity not in self.buffered_cameras:
                raise ServiceValidationError(
                    f"Lookback ({lookback_val}s) was requested for '{camera_entity}', but this camera is not configured for stream buffering in LLM Vision Settings. "
                    "Please add it under LLM Vision Settings -> Stream Buffer Cameras."
                )

            if lookback_val > self.max_buffer_seconds:
                raise ServiceValidationError(
                    f"Requested lookback ({lookback_val}s) exceeds the configured maximum requested lookback ({self.max_buffer_seconds}s) in LLM Vision Settings. "
                    "Please increase the buffer limit in Settings or reduce lookback."
                )

            lease = self.leases.get(camera_entity)
            if lease is None or lease.stream is None:
                raise ServiceValidationError(
                    f"Stream buffer is not active for camera '{camera_entity}'. "
                    "The stream keeper has not acquired a live stream."
                )

            # Check stream availability
            if hasattr(lease.stream, "available"):
                avail = (
                    lease.stream.available()
                    if callable(lease.stream.available)
                    else lease.stream.available
                )
                if not avail:
                    raise ServiceValidationError(
                        f"Camera stream for '{camera_entity}' is currently unavailable."
                    )

            # Retrieve CURRENT native output and validate identity against lease
            outputs = (
                lease.stream.outputs()
                if callable(lease.stream.outputs)
                else getattr(lease.stream, "outputs", {})
            )
            current_hls = None
            if isinstance(outputs, (dict, Mapping)) or hasattr(outputs, "get"):
                current_hls = outputs.get("hls")

            if current_hls is None or current_hls is not lease.provider:
                raise ServiceValidationError(
                    f"HLS stream output for camera '{camera_entity}' is missing or has been replaced."
                )

            # Retrieve segments
            segments = []
            if hasattr(current_hls, "get_segments"):
                raw_segs = current_hls.get_segments()
                segments = list(raw_segs) if raw_segs is not None else []

            if not segments:
                raise ServiceValidationError(
                    f"Stream buffer for '{camera_entity}' is empty (warming up). No buffered segments are currently available."
                )

            # Core async_record prepends completed segments excluding current in-progress segment:
            # num_segments = min(int(lookback / hls.target_duration) + 1, MAX_SEGMENTS)
            # prepended = list(hls.get_segments())[-num_segments - 1 : -1]
            target_duration = float(
                getattr(current_hls, "target_duration", 6.0) or 6.0
            )
            selected_count = min(
                int(lookback_val / target_duration) + 1, max_segments
            )

            # Slice exact prepended segments from Core formula
            eligible_segments = list(segments)[-selected_count - 1 : -1]
            if not eligible_segments:
                raise ServiceValidationError(
                    f"Stream buffer for '{camera_entity}' is empty or warming up (no completed segments available)."
                )

            for seg in eligible_segments:
                if getattr(seg, "duration", 0) <= 0:
                    raise ServiceValidationError(
                        f"Stream buffer for '{camera_entity}' contains in-progress or incomplete segment in eligible lookback history."
                    )

            # Duration check on actual prepended segments selected by Core
            available_duration = sum(
                float(getattr(s, "duration", 0)) for s in eligible_segments
            )
            if available_duration < lookback_val:
                raise ServiceValidationError(
                    f"Insufficient buffered history for camera '{camera_entity}': requested {lookback_val}s, but only {available_duration:.1f}s of completed footage is available across Core's {len(eligible_segments)} prepended segments (stream warming up). "
                    "Please allow the buffer to accumulate longer segments or reduce the requested lookback."
                )

            # Stream ID continuity check against latest current segment (discontinuity crossing)
            current_segment = segments[-1]
            current_stream_id = getattr(current_segment, "stream_id", None)
            if current_stream_id is not None:
                for seg in eligible_segments:
                    if getattr(seg, "stream_id", None) != current_stream_id:
                        raise ServiceValidationError(
                            f"Stream buffer for '{camera_entity}' crossed a stream restart boundary (stream_id mismatch between history and current stream)."
                        )

            # Sequence continuity check
            if len(eligible_segments) > 1 and hasattr(
                eligible_segments[0], "sequence"
            ):
                for idx in range(len(eligible_segments) - 1):
                    curr_seq = getattr(eligible_segments[idx], "sequence", None)
                    next_seq = getattr(eligible_segments[idx + 1], "sequence", None)
                    if (
                        curr_seq is not None
                        and next_seq is not None
                        and next_seq != curr_seq + 1
                    ):
                        raise ServiceValidationError(
                            f"Stream buffer for '{camera_entity}' contains non-contiguous segment sequence (gap in buffer history)."
                        )

            # Absolute timestamp freshness check using dt_util.utcnow() on both history and current segment
            now = dt_util.utcnow()
            for seg_to_check in (eligible_segments[-1], current_segment):
                start_time = getattr(seg_to_check, "start_time", None)
                if start_time is not None:
                    if isinstance(start_time, datetime):
                        if start_time.tzinfo is None:
                            start_time = start_time.replace(tzinfo=dt_util.UTC)
                        age = (now - start_time).total_seconds()
                        seg_dur = float(
                            getattr(
                                seg_to_check, "duration", target_duration
                            )
                            or target_duration
                        )
                        max_allowed_age = seg_dur + (target_duration * 1.5)
                        if age > max_allowed_age:
                            raise ServiceValidationError(
                                f"Stream buffer for '{camera_entity}' contains stale footage (segment started {age:.1f}s ago; tolerance is {max_allowed_age:.1f}s). Stream may be stalled."
                            )

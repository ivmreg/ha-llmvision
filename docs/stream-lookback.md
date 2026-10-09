# Stream Lookback & Buffer Guide

The Stream Lookback feature enables [`llmvision.stream_analyzer`](https://github.com/valentinfrlch/ha-llmvision) to inspect frames leading up to an event by borrowing Home Assistant Core's native HLS pre-roll buffer. Target and test requirements pin [Home Assistant Core 2025.11.2](https://github.com/home-assistant/core/tree/2025.11.2/homeassistant/components/stream).

## Configuration & Usage

1. **Enable Cameras**: In **Settings -> Devices & Services -> LLM Vision -> Reconfigure Settings**, configure:
   - **Stream buffer cameras**: Select camera entities to keep continuously pre-buffered.
   - **Maximum requested lookback** (`stream_buffer_seconds`): GUI setting range 5–60s defining the upper ceiling for requests. Service calls request `lookback` from 0 up to this configured maximum (`lookback: 0` disables pre-roll). *Note: This setting validates service call input; it does not resize or reconfigure Core's underlying stream buffer.*
2. **Service Call Example**:
   Specify your configured vision model provider entry ID (e.g., OpenAI, Anthropic, or Google) for `provider`, not the Settings entry.
   ```yaml
   action: llmvision.stream_analyzer
   data:
     provider: "<vision_provider_entry_id>"
     image_entity: camera.front_door
     message: "Describe what happened right before motion occurred."
     lookback: 10
     expose_images: true
     store_in_timeline: true
   ```

## Actual Core Semantics & Buffer Mechanics

- **Core Prepending**: When lookback is requested, Core prepends completed segments:
  $$\text{selected} = \min\left(\left\lfloor\frac{\text{lookback}}{\text{target\_duration}}\right\rfloor + 1, \text{MAX\_SEGMENTS}\right)$$
  using `hls.get_segments()[-selected - 1 : -1]` (excluding the latest segment in progress).
- **No Guaranteed Duration**: Core caps its ring buffer at `MAX_SEGMENTS = 5` ([`const.py`](https://github.com/home-assistant/core/blob/2025.11.2/homeassistant/components/stream/const.py)). For example, with four completed 6s segments and one in-progress segment, up to 24s of completed footage may be available, but this is only an illustrative example—actual duration depends on camera keyframe intervals, segment timing, and Core LL-HLS configuration. Because Core awaits the next segment during recording, pre-check verification is best-effort and cannot guarantee exact second-level duration.
- **Warmup & Continuity Guard**: Positive lookback calls verify active stream leases, current output identity, contiguous segment sequences, stream ID continuity, and segment timestamp freshness. If the buffer is warming up, stale, or insufficient, `ServiceValidationError` is raised explicitly without silent fallback to post-event snapshots.

## Lifecycle & Asset Handling

- **Shared Stream Safety**: LLM Vision preserves original `preload_stream` settings. If originally `False`, `preload_stream` is restored to `False` on entry unload and `provider.idle_timer.start()` is called to allow native Core idle expiry. Shared HLS providers and active viewers are never stopped or cancelled.
- **Unload Serialization**: Manager startup, state changes, and unloads are serialized with `asyncio.Lock` to prevent dropped coroutines, late reacquisition, or orphaned leases.
- **Event Storage & Assets**: Saved frames use sanitized filenames and are committed to disk before bookkeeping. When stored in Timeline, frame paths persist in the database using schema version 5 while preserving legacy records. *Note: The current LLM Vision dashboard card displays the keyframe; saved frames are retained in the event record and returned in the service response.*

## Hardware Considerations & Acceptance Steps

- **Resource Impact**: Continuously decoding camera streams consumes CPU and network bandwidth. Monitor host hardware usage when enabling multiple buffered cameras.
- **Hardware Acceptance Test**:
  1. Verify lookback across both cases: with a live camera viewer open, or after integration reload (wait for the stream buffer to warm up after reload before triggering movement; do not assume a fixed wait time).
  2. Trigger movement in front of the configured camera.
  3. Call `llmvision.stream_analyzer` with positive `lookback`.
  4. Verify the service response contains valid `saved_frames` and `key_frame` paths representing pre-roll motion.
*(Note: Automated test suites validate API contracts and mock shapes; no physical camera or hardware acceptance testing has been completed in this environment.)*

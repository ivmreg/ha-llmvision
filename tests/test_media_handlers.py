"""Unit tests for media_handlers.py module."""
import asyncio
import os
import io
import base64
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
from types import SimpleNamespace
from homeassistant.exceptions import ServiceValidationError
from custom_components.llmvision.const import DOMAIN
from custom_components.llmvision.media_handlers import MediaProcessor


def _make_jpeg_bytes(color):
    """Create JPEG bytes for a solid-color test image."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


class _AsyncResponseContext:
    """Minimal async context manager for aiohttp-style responses."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _StreamingContent:
    """Streaming response content stub."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncWriter:
    """Async file writer compatible with aiofile.async_open usage."""

    def __init__(self, path):
        self._path = path
        self._file = None

    async def __aenter__(self):
        self._file = open(self._path, "wb")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._file is not None:
            self._file.close()
        return False

    async def write(self, data):
        if self._file is None:
            raise RuntimeError("Writer not opened")
        self._file.write(data)


class TestMediaProcessor:
    """Test MediaProcessor class."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = Mock()
        hass.data = {}
        hass.loop = Mock()
        hass.loop.run_in_executor = AsyncMock()
        hass.states = Mock()
        hass.states.get = Mock()
        hass.config = Mock()
        hass.config.path = Mock(return_value="/mock/path")
        return hass

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        client = Mock()
        client.add_frame = Mock()
        return client

    @pytest.fixture
    def processor(self, mock_hass, mock_client, tmp_path):
        """Create a MediaProcessor instance."""
        with patch('custom_components.llmvision.media_handlers.async_get_clientsession'):
            proc = MediaProcessor(mock_hass, mock_client)
            proc.snapshots_path = str(tmp_path / "snapshots") + "/"
            return proc

    def test_init(self, mock_hass, mock_client):
        """Test MediaProcessor initialization."""
        with patch('custom_components.llmvision.media_handlers.async_get_clientsession'):
            processor = MediaProcessor(mock_hass, mock_client)
        assert processor.hass == mock_hass
        assert processor.client == mock_client
        assert processor.base64_images == []
        assert processor.filenames == []
        assert processor.key_frame == ""
        assert processor.snapshots_path == "/media/llmvision/snapshots/"

    @pytest.mark.asyncio
    async def test_encode_image(self, processor):
        """Test _encode_image method."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')

        result = await processor._encode_image(img)

        assert isinstance(result, str)
        assert len(result) > 0
        # Verify it's valid base64
        base64.b64decode(result)

    @pytest.mark.asyncio
    async def test_save_clip_writes_base64_image(self, processor, tmp_path):
        """_save_clip should decode base64 image payloads before writing."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        output = tmp_path / "snapshot.jpg"

        with patch("custom_components.llmvision.media_handlers.os.makedirs"):
            await processor._save_clip(
                image_data=base64.b64encode(b"image-bytes").decode("utf-8"),
                image_path=str(output),
            )

        assert output.read_bytes() == b"image-bytes"

    @pytest.mark.asyncio
    async def test_expose_image_uses_frame_path_once(self, processor, tmp_path):
        """_expose_image should derive image data from a file and track all saved frames."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        source = tmp_path / "frame.jpg"
        source.write_bytes(_make_jpeg_bytes("navy"))
        processor._save_clip = AsyncMock()

        with patch("custom_components.llmvision.media_handlers.os.makedirs"):
            await processor._expose_image(
                frame_name="7",
                image_data=None,
                uid="deadbeef",
                frame_path=str(source),
            )
            await processor._expose_image(
                frame_name="8",
                image_data="ignored",
                uid="second",
            )

        assert processor.key_frame.endswith("deadbeef-7.jpg")
        assert len(processor.saved_frames) == 2
        assert processor.saved_frames[0].endswith("deadbeef-7.jpg")
        assert processor.saved_frames[1].endswith("second-8.jpg")
        assert processor._save_clip.await_count == 2

    @pytest.mark.asyncio
    async def test_select_keyframe_index_picks_lowest_similarity(self, processor):
        """_select_keyframe_index should return the most dissimilar candidate."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        processor._similarity_score = Mock(side_effect=[0.8, 0.2, 0.5])

        result = await processor._select_keyframe_index(
            _make_jpeg_bytes("black"),
            [
                _make_jpeg_bytes("gray"),
                _make_jpeg_bytes("white"),
                _make_jpeg_bytes("red"),
            ],
        )

        assert result == 1

    def test_convert_to_rgb_rgba(self, processor):
        """Test _convert_to_rgb with RGBA image."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))

        result = processor._convert_to_rgb(img)

        assert result.mode == 'RGB'

    def test_convert_to_rgb_already_rgb(self, processor):
        """Test _convert_to_rgb with RGB image."""
        img = Image.new('RGB', (100, 100), color='red')

        result = processor._convert_to_rgb(img)

        assert result.mode == 'RGB'

    @pytest.mark.asyncio
    async def test_resize_image_with_img(self, processor):
        """Test resize_image with PIL Image object."""
        # Create a test image
        img = Image.new('RGB', (200, 100), color='blue')

        result = await processor.resize_image(
            target_width=100,
            img=img
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_with_image_path(self, processor, tmp_path):
        """resize_image should support file-based inputs."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        image_path = tmp_path / "photo.jpg"
        image_path.write_bytes(_make_jpeg_bytes("green"))

        result = await processor.resize_image(
            target_width=8, image_path=str(image_path)
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_resize_image_requires_input(self, processor):
        """resize_image should reject empty inputs."""
        with pytest.raises(ServiceValidationError, match="No image data provided"):
            await processor.resize_image(target_width=10)

    def test_similarity_score(self, processor):
        """Test _similarity_score method."""
        # Create two similar images
        img1 = Image.new('L', (100, 100), color=128)
        img2 = Image.new('L', (100, 100), color=130)

        score = processor._similarity_score(img1, img2)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_fetch_retries_until_success(self, processor):
        """_fetch should retry failed responses before succeeding."""
        first = SimpleNamespace(ok=False, status=500)
        second = SimpleNamespace(
            ok=True, status=200, read=AsyncMock(return_value=b"ok")
        )
        processor.session.get.side_effect = [
            _AsyncResponseContext(first),
            _AsyncResponseContext(second),
        ]

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            AsyncMock(),
        ):
            result = await processor._fetch("http://camera/frame.jpg", retry_delay=0)

        assert result == b"ok"

    @pytest.mark.asyncio
    async def test_fetch_streams_to_target_file(self, processor, tmp_path):
        """_fetch should stream large responses to disk when target_file is provided."""
        target = tmp_path / "clip.mp4"
        response = SimpleNamespace(
            ok=True,
            status=200,
            content=_StreamingContent([b"abc", b"def"]),
        )
        processor.session.get.return_value = _AsyncResponseContext(response)

        with patch(
            "custom_components.llmvision.media_handlers.async_open",
            side_effect=lambda path, mode: _FakeAsyncWriter(path),
        ):
            result = await processor._fetch(
                "http://camera/clip.mp4",
                target_file=str(target),
            )

        assert result is None
        assert target.read_bytes() == b"abcdef"

    @pytest.mark.asyncio
    async def test_fetch_returns_none_after_exhausted_errors(self, processor):
        """_fetch should return None after repeated exceptions."""
        processor.session.get.side_effect = RuntimeError("boom")

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            AsyncMock(),
        ):
            result = await processor._fetch(
                "http://camera/frame.jpg",
                max_retries=2,
                retry_delay=0,
                entity_name="front",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_record_returns_frames_in_capture_order(self, processor):
        """Selected stream frames should be emitted in capture order."""
        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        processor.hass.states.get.return_value = SimpleNamespace(
            attributes={"entity_picture": "/api/camera_proxy/camera.front"}
        )
        frame_iter = iter(
            [
                _make_jpeg_bytes("black"),
                _make_jpeg_bytes("gray"),
                _make_jpeg_bytes("white"),
            ]
        )

        async def fake_fetch(*args, **kwargs):
            frame = next(frame_iter, None)
            if frame is None:
                clock["now"] = 10.0
            return frame

        processor._fetch = AsyncMock(side_effect=fake_fetch)
        processor._similarity_score = Mock(side_effect=[0.9, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=["encoded-0", "encoded-1", "encoded-2"]
        )
        processor._select_keyframe_index = AsyncMock(return_value=0)

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=lambda: clock["now"],
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            await processor.record(
                image_entities=["camera.front"],
                duration=2.5,
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert [
            call.kwargs["filename"]
            for call in processor.client.add_frame.call_args_list
        ] == ["camera0-frame-0", "camera0-frame-1", "camera0-frame-2"]
        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["encoded-0", "encoded-1", "encoded-2"]

    @pytest.mark.asyncio
    async def test_record_raises_when_no_cameras_available(self, processor):
        """record should fail when all cameras are unavailable."""
        clock = {"now": 0.0}

        async def fake_sleep(delay):
            clock["now"] += delay

        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.time.time",
            side_effect=lambda: clock["now"],
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.sleep",
            side_effect=fake_sleep,
        ), pytest.raises(
            ServiceValidationError, match="No cameras available"
        ):
            await processor.record(
                image_entities=["camera.front"],
                duration=0.1,
                max_frames=1,
                target_width=64,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_supports_entities_and_paths(self, processor, tmp_path):
        """add_images should add frames from both camera entities and file paths."""
        image_path = tmp_path / "still.jpg"
        image_path.write_bytes(_make_jpeg_bytes("teal"))
        entity_state = SimpleNamespace(
            attributes={
                "entity_picture": "/api/camera_proxy/camera.front",
                "friendly_name": "Front Door",
            }
        )
        processor.hass.states.get.return_value = entity_state
        processor._fetch = AsyncMock(return_value=b"entity-bytes")
        processor.resize_image = AsyncMock(side_effect=["entity-frame", "path-frame"])
        processor._expose_image = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[str(image_path)],
                target_width=128,
                include_filename=True,
                expose_images=True,
            )

        assert result is processor.client
        assert [
            call.kwargs["filename"]
            for call in processor.client.add_frame.call_args_list
        ] == ["Front Door", "still"]
        assert processor._expose_image.await_count == 2

    @pytest.mark.asyncio
    async def test_add_images_raises_for_missing_file(self, processor):
        """add_images should reject missing local image files."""
        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="does not exist"):
            await processor.add_images(
                image_entities=[],
                image_paths=["/missing/file.jpg"],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_raises_when_all_cameras_unavailable(self, processor):
        """add_images should fail when no camera image can be obtained."""
        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="No cameras available"):
            await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_images_wraps_attribute_errors(self, processor):
        """add_images should surface malformed entity state access as validation errors."""
        processor.hass.states.get.return_value = object()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="Error accessing camera entity"):
            await processor.add_images(
                image_entities=["camera.front"],
                image_paths=[],
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_returns_selected_frames_in_capture_order(self, processor):
        """Selected video keyframes should be added in capture order."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        frame_bytes = [
            _make_jpeg_bytes("black"),
            _make_jpeg_bytes("gray"),
            _make_jpeg_bytes("white"),
        ]
        encoded_by_frame = {
            frame_bytes[0]: "encoded-0",
            frame_bytes[1]: "encoded-1",
            frame_bytes[2]: "encoded-2",
        }

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self, payload):
                self.pid = 1234
                self.stdout = FakeStdout([payload, b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        processor._similarity_score = Mock(side_effect=[0.9, 0.1])
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kwargs: encoded_by_frame[
                image_data or b""
            ]
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess(b"".join(frame_bytes))),
        ):
            await processor.add_video(
                video_path="/tmp/test_clip.mp4",
                base_url="http://ha.local",
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert [
            call.kwargs["base64_image"]
            for call in processor.client.add_frame.call_args_list
        ] == ["encoded-0", "encoded-1", "encoded-2"]

    @pytest.mark.asyncio
    async def test_add_video_signs_api_path_and_exposes_keyframe(self, processor):
        """add_video should sign local API paths and expose the selected keyframe."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        frame_bytes = [_make_jpeg_bytes("black")]
        temp_video = "/tmp/signed-video.mp4"

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 55
                self.stdout = FakeStdout([b"".join(frame_bytes), b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        class FakeTempFile:
            name = temp_video

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        processor.resize_image = AsyncMock(return_value="encoded-frame")
        processor._select_keyframe_index = AsyncMock(return_value=0)
        processor._expose_image = AsyncMock()
        processor._fetch = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.async_sign_path",
            return_value="/api/signed/path?authSig=1",
        ), patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(),
        ), patch(
            "custom_components.llmvision.media_handlers.os.path.exists",
            side_effect=lambda path: path == temp_video,
        ), patch(
            "custom_components.llmvision.media_handlers.os.path.getsize",
            return_value=10,
        ), patch(
            "custom_components.llmvision.media_handlers.os.remove",
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ):
            await processor.add_video(
                video_path="/api/frigate/notifications/test/clip.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=True,
                expose_images=True,
            )

        assert (
            processor.client.add_frame.call_args.kwargs["filename"]
            == "path?authSig=1 (frame 1)"
        )
        processor._expose_image.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_video_raises_when_http_download_is_empty(
        self, processor, tmp_path
    ):
        """add_video should fail when the downloaded remote clip is empty."""
        temp_video = tmp_path / "download.mp4"

        class FakeTempFile:
            def __init__(self, path):
                self.name = str(path)

            def __enter__(self):
                temp_video.write_bytes(b"")
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        processor._fetch = AsyncMock(return_value=None)

        with patch(
            "custom_components.llmvision.media_handlers.tempfile.NamedTemporaryFile",
            return_value=FakeTempFile(temp_video),
        ), pytest.raises(ServiceValidationError, match="Failed to fetch video"):
            await processor.add_video(
                video_path="http://example.com/clip.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_raises_when_no_frames_are_extracted(self, processor):
        """add_video should fail cleanly if ffmpeg yields no JPEG frames."""

        class FakeStdout:
            async def read(self, _size):
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 44
                self.stdout = FakeStdout()
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ), pytest.raises(ServiceValidationError, match="No frames extracted"):
            await processor.add_video(
                video_path="/tmp/empty.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_video_raises_when_stdout_is_missing(self, processor):
        """add_video should reject ffmpeg processes without a stdout pipe."""

        class FakeProcess:
            def __init__(self):
                self.pid = 77
                self.stdout = None
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ), pytest.raises(ServiceValidationError, match="stdout not available"):
            await processor.add_video(
                video_path="/tmp/no-stdout.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

    @pytest.mark.asyncio
    async def test_add_videos_processes_event_ids(self, processor):
        """add_videos should convert event ids into Frigate clip URLs."""
        processor.add_video = AsyncMock()

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor.add_videos(
                video_paths=["/tmp/local.mp4"],
                event_ids=["abc123"],
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert result is processor.client
        assert processor.add_video.await_count == 2
        assert any(
            call.kwargs["video_path"] == "/api/frigate/notifications/abc123/clip.mp4"
            for call in processor.add_video.await_args_list
        )

    @pytest.mark.asyncio
    async def test_add_streams_delegates_to_record(self, processor):
        """add_streams should proxy to record when image entities are present."""
        processor.record = AsyncMock()

        result = await processor.add_streams(
            image_entities=["camera.front"],
            duration=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

        assert result is processor.client
        processor.record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_streams_with_lookback_delegates_to_record(
        self, processor
    ):
        """add_streams should call record passing lookback parameter."""
        processor.record = AsyncMock()

        result = await processor.add_streams(
            image_entities=["camera.front"],
            duration=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
            lookback=5,
        )

        assert result is processor.client
        processor.record.assert_awaited_once_with(
            image_entities=["camera.front"],
            duration=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
            lookback=5,
        )

    @pytest.mark.asyncio
    async def test_record_with_lookback_delegates_to_helper(self, processor):
        """record should delegate to _record_with_lookback when lookback > 0."""
        processor._record_with_lookback = AsyncMock(return_value=processor.client)

        result = await processor.record(
            image_entities=["camera.front"],
            duration=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
            lookback=5,
        )

        assert result is processor.client
        processor._record_with_lookback.assert_awaited_once_with(
            image_entities=["camera.front"],
            duration=5,
            lookback=5,
            max_frames=2,
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

    @pytest.mark.asyncio
    async def test_record_with_lookback_success(self, processor, tmp_path):
        """_record_with_lookback should call camera.record service and add_video."""
        processor.hass.states.get.return_value = SimpleNamespace(
            attributes={"friendly_name": "Front Porch"}
        )
        processor.hass.services = Mock()
        processor.hass.services.async_call = AsyncMock()
        processor.add_video = AsyncMock()

        # Mock run_in_executor to simulate file existing with size
        async def mock_run_in_executor(executor, func, *args):
            if func == os.path.exists:
                return True
            if func == os.path.getsize:
                return 1024
            return func(*args)

        processor.hass.loop.run_in_executor = mock_run_in_executor

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ):
            result = await processor._record_with_lookback(
                image_entities=["camera.front_porch"],
                duration=5,
                lookback=5,
                max_frames=3,
                target_width=1280,
                include_filename=True,
                expose_images=True,
            )

        assert result is processor.client
        processor.hass.services.async_call.assert_awaited_once()
        call_domain, call_service, call_data = processor.hass.services.async_call.call_args[0]
        assert call_domain == "camera"
        assert call_service == "record"
        assert call_data["entity_id"] == "camera.front_porch"
        assert call_data["duration"] == 5
        assert call_data["lookback"] == 5
        processor.add_video.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_with_positive_lookback_failure_raises_never_falls_back(self, processor):
        """record should raise ServiceValidationError when lookback > 0 fails, never falling back to snapshot loop."""
        processor._record_with_lookback = AsyncMock(
            side_effect=ServiceValidationError("Stream unreachable")
        )
        processor.hass.states.get = Mock()

        with pytest.raises(ServiceValidationError, match="Stream unreachable"):
            await processor.record(
                image_entities=["camera.front_porch"],
                duration=1,
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
                lookback=5,
            )

        processor._record_with_lookback.assert_awaited_once()
        # Verify that snapshot polling was never attempted
        assert processor.hass.states.get.call_count == 0

    @pytest.mark.asyncio
    async def test_add_visual_data_delegates_to_add_images(self, processor):
        """add_visual_data should proxy to add_images."""
        processor.add_images = AsyncMock()

        result = await processor.add_visual_data(
            image_entities=["camera.front"],
            image_paths=["/tmp/still.jpg"],
            target_width=128,
            include_filename=False,
            expose_images=False,
        )

        assert result is processor.client
        processor.add_images.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_record_lookback_sibling_cancellation_and_cleanup_on_error(
        self, processor, tmp_path
    ):
        """Multi-camera lookback must cancel and join sibling tasks on early error and cleanup temp files."""
        processor.snapshots_path = str(tmp_path / "snapshots") + "/"
        cam2_cancelled = False

        processor.hass.states.get.side_effect = lambda entity: SimpleNamespace(
            attributes={"friendly_name": entity}
        )

        async def fake_async_call(domain, service, data, blocking=True):
            entity = data["entity_id"]
            if entity == "camera.cam1":
                raise ServiceValidationError("Camera 1 stream unreachable")
            elif entity == "camera.cam2":
                nonlocal cam2_cancelled
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cam2_cancelled = True
                    raise

        processor.hass.services = Mock()
        processor.hass.services.async_call = fake_async_call

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="Camera 1 stream unreachable"):
            await processor.record(
                image_entities=["camera.cam1", "camera.cam2"],
                duration=2,
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
                lookback=5,
            )

        # Sibling task for Camera 2 must have been cancelled
        assert cam2_cancelled is True
        # No temp files remaining in snapshots_path
        if os.path.exists(processor.snapshots_path):
            temp_files = [
                f for f in os.listdir(processor.snapshots_path) if f.startswith("temp_lookback_")
            ]
            assert len(temp_files) == 0

    @pytest.mark.asyncio
    async def test_add_videos_deterministic_input_order_and_keyframe(
        self, processor, tmp_path
    ):
        """add_videos must execute real add_video sequentially in input order and resolve keyframes per-video locally."""
        processor.snapshots_path = str(tmp_path / "snapshots") + "/"
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )

        v1_frames = [_make_jpeg_bytes((10, 10, i * 50)) for i in range(3)]
        v2_frames = [_make_jpeg_bytes((200, 200, i * 50)) for i in range(2)]

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self, frames):
                self.pid = 101
                self.stdout = FakeStdout([b"".join(frames), b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        async def fake_create_subprocess_exec(*args, **kwargs):
            cmd_str = " ".join(args)
            if "clip1.mp4" in cmd_str:
                return FakeProcess(v1_frames)
            else:
                return FakeProcess(v2_frames)

        processor._select_keyframe_index = AsyncMock(side_effect=[1, 0])
        processor._similarity_score = Mock(return_value=0.5)
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kw: base64.b64encode(image_data or b"").decode("utf-8")
        )

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            fake_create_subprocess_exec,
        ):
            await processor.add_videos(
                video_paths=["/media/clip1.mp4", "/media/clip2.mp4"],
                event_ids=[],
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=True,
            )

        assert len(processor.saved_frames) == 5
        assert all("clip1-frame_" in f for f in processor.saved_frames[:3])
        assert all("clip2-frame_" in f for f in processor.saved_frames[3:])

        clip1_keyframe = processor.saved_frames[1]
        assert "clip1-frame_2" in clip1_keyframe

        clip2_keyframe = processor.saved_frames[3]
        assert "clip2-frame_1" in clip2_keyframe
        assert processor.key_frame == clip2_keyframe

    @pytest.mark.asyncio
    async def test_path_separators_sanitization_and_write_failure_state(
        self, processor, tmp_path
    ):
        """Friendly names must keep readable LLM labels, while filesystem paths sanitize all separators; write failures must not alter bookkeeping."""
        processor.snapshots_path = str(tmp_path / "snapshots") + "/"
        encoded_img = base64.b64encode(b"fake-image").decode("utf-8")

        frame_bytes = [_make_jpeg_bytes("blue")]

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 99
                self.stdout = FakeStdout([b"".join(frame_bytes), b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        processor.resize_image = AsyncMock(return_value=encoded_img)
        processor._select_keyframe_index = AsyncMock(return_value=0)
        exposed_names = []

        real_expose = processor._expose_image

        async def track_expose(frame_name, image_data, uid, frame_path=None):
            exposed_names.append(frame_name)
            return await real_expose(frame_name, image_data, uid, frame_path=frame_path)

        processor._expose_image = track_expose

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ):
            await processor.add_video(
                video_path="/media/test.mp4",
                base_url="http://ha.local",
                max_frames=1,
                target_width=128,
                include_filename=True,
                expose_images=True,
                override_filename="Front/Porch/Cam:1\\2",
            )

        assert (
            processor.client.add_frame.call_args.kwargs["filename"]
            == "Front/Porch/Cam:1\\2 (frame 1)"
        )
        assert len(exposed_names) == 1
        assert "/" not in exposed_names[0]
        assert "\\" not in exposed_names[0]
        assert ":" not in exposed_names[0]
        assert "Front_Porch_Cam_1_2" in exposed_names[0]

        processor.saved_frames.clear()
        processor.key_frame = ""
        processor._save_clip = AsyncMock(side_effect=IOError("Disk write failed"))

        with pytest.raises(IOError, match="Disk write failed"):
            await real_expose("failed_frame", encoded_img, "bad12345")

        assert len(processor.saved_frames) == 0
        assert processor.key_frame == ""

    @pytest.mark.asyncio
    async def test_late_motion_under_bounded_candidates(self, processor):
        """Candidate frame bounding must drain entire clip to EOF (80+ frames), bound heap to cap 30, and select late motion frames."""
        processor.hass.loop.run_in_executor.side_effect = (
            lambda _executor, func, *args: func(*args)
        )
        total_frames = 85
        frame_bytes = [
            _make_jpeg_bytes((i * 3 % 256, (i * 7 + 10) % 256, (i * 11 + 20) % 256))
            for i in range(total_frames)
        ]

        class FakeStdout:
            def __init__(self, chunks):
                self._chunks = list(chunks)

            async def read(self, _size):
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

        class FakeProcess:
            def __init__(self):
                self.pid = 88
                self.stdout = FakeStdout([b"".join(frame_bytes), b""])
                self.stderr = None
                self.returncode = 0

            async def wait(self):
                return self.returncode

        scores = [0.95] * 60 + [0.05 + (i * 0.005) for i in range(24)]
        processor._similarity_score = Mock(side_effect=scores)

        added_frames = []
        processor.client.add_frame = Mock(
            side_effect=lambda base64_image, filename: added_frames.append((base64_image, filename))
        )
        processor.resize_image = AsyncMock(
            side_effect=lambda target_width, image_data=None, **kw: base64.b64encode(image_data or b"").decode("utf-8")
        )

        with patch(
            "custom_components.llmvision.media_handlers.asyncio.create_subprocess_exec",
            AsyncMock(return_value=FakeProcess()),
        ):
            await processor.add_video(
                video_path="/media/long_clip.mp4",
                base_url="http://ha.local",
                max_frames=3,
                target_width=128,
                include_filename=False,
                expose_images=False,
            )

        assert processor._similarity_score.call_count == 84
        assert len(added_frames) == 3

        first_encoded = base64.b64encode(frame_bytes[0]).decode("utf-8")
        assert added_frames[0][0] == first_encoded
        assert added_frames[0][1] == "Video frame 1"

        late_encoded = [base64.b64encode(b).decode("utf-8") for b in frame_bytes[61:]]
        assert added_frames[1][0] in late_encoded
        assert added_frames[2][0] in late_encoded

    @pytest.mark.asyncio
    async def test_record_zero_lookback_unchanged(self, processor):
        """record with lookback=0 must execute standard snapshot polling and never call _record_with_lookback."""
        processor._record_with_lookback = AsyncMock()
        processor.hass.states.get.return_value = None

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="No cameras available"):
            await processor.record(
                image_entities=["camera.front"],
                duration=1,
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
                lookback=0,
            )

        processor._record_with_lookback.assert_not_called()

    @pytest.mark.asyncio
    async def test_lookback_cancellation_releases_only_matching_owned_recorder(
        self, processor, tmp_path
    ):
        """When cancelled, only the matching owned RecorderOutput provider is removed; HLS and stream remain intact."""
        processor.snapshots_path = str(tmp_path / "snapshots") + "/"

        mock_recorder = SimpleNamespace(name="recorder", video_path="")
        mock_hls = SimpleNamespace(name="hls")
        mock_stream = Mock()
        mock_stream.outputs = Mock(return_value={"recorder": mock_recorder, "hls": mock_hls})
        mock_stream.remove_provider = AsyncMock()
        mock_stream.stop = Mock()

        processor.hass.data[DOMAIN] = {"active_streams": {"camera.cam2": mock_stream}}
        processor.hass.states.get.side_effect = lambda entity: SimpleNamespace(
            attributes={"friendly_name": entity}
        )

        async def fake_async_call(domain, service, data, blocking=True):
            entity = data["entity_id"]
            if entity == "camera.cam1":
                raise ServiceValidationError("Camera 1 stream unreachable")
            elif entity == "camera.cam2":
                mock_recorder.video_path = data["filename"]
                await asyncio.sleep(10.0)

        processor.hass.services = Mock()
        processor.hass.services.async_call = fake_async_call

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="Camera 1 stream unreachable"):
            await processor.record(
                image_entities=["camera.cam1", "camera.cam2"],
                duration=2,
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
                lookback=5,
            )

        mock_stream.remove_provider.assert_awaited_once_with(mock_recorder)
        assert mock_hls not in [call.args[0] for call in mock_stream.remove_provider.await_args_list]
        mock_stream.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_lookback_cancellation_preserves_borrowed_mismatched_recorder(
        self, processor, tmp_path
    ):
        """When cancelled, mismatched/borrowed recorder provider is NEVER removed."""
        processor.snapshots_path = str(tmp_path / "snapshots") + "/"

        borrowed_recorder = SimpleNamespace(name="recorder", video_path="/other/recording.mp4")
        mock_stream = Mock()
        mock_stream.outputs = Mock(return_value={"recorder": borrowed_recorder})
        mock_stream.remove_provider = AsyncMock()
        mock_stream.stop = Mock()

        processor.hass.data[DOMAIN] = {"active_streams": {"camera.cam2": mock_stream}}
        processor.hass.states.get.side_effect = lambda entity: SimpleNamespace(
            attributes={"friendly_name": entity}
        )

        async def fake_async_call(domain, service, data, blocking=True):
            entity = data["entity_id"]
            if entity == "camera.cam1":
                raise ServiceValidationError("Camera 1 failed")
            elif entity == "camera.cam2":
                await asyncio.sleep(10.0)

        processor.hass.services = Mock()
        processor.hass.services.async_call = fake_async_call

        with patch(
            "custom_components.llmvision.media_handlers.get_url",
            return_value="http://ha.local",
        ), pytest.raises(ServiceValidationError, match="Camera 1 failed"):
            await processor.record(
                image_entities=["camera.cam1", "camera.cam2"],
                duration=2,
                max_frames=2,
                target_width=128,
                include_filename=False,
                expose_images=False,
                lookback=5,
            )

        mock_stream.remove_provider.assert_not_called()
        mock_stream.stop.assert_not_called()

import base64
import io
import os
import uuid
import logging
import time
import asyncio
import tempfile
import heapq
import re
from aiofile import async_open
from datetime import timedelta
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.components.http.auth import async_sign_path
from homeassistant.components.media_source import is_media_source_id
from homeassistant.components.media_player.browse_media import (
    async_process_play_media_url,
)

from urllib.parse import urlparse
from functools import partial
from PIL import Image, UnidentifiedImageError
import numpy as np
from homeassistant.helpers.network import get_url
from homeassistant.exceptions import ServiceValidationError

from .const import DOMAIN, MAX_KEYFRAME_CANDIDATES

_LOGGER = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Sanitize name for safe filesystem path usage, preventing directory traversal and separator issues."""
    if not name:
        return "clip"
    sanitized = name.replace("\x00", "")
    sanitized = re.sub(r"[\\/:]+", "_", sanitized)
    sanitized = re.sub(r"[^\w\-.]", "_", sanitized)
    sanitized = re.sub(r"\.{2,}", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("._")
    if len(sanitized) > 64:
        sanitized = sanitized[:64].rstrip("._")
    return sanitized or "clip"


class MediaProcessor:
    def __init__(self, hass, client):
        self.hass = hass
        self.session = async_get_clientsession(self.hass)
        self.client = client
        self.base64_images = []
        self.filenames = []
        self.snapshots_path = f"/media/{DOMAIN}/snapshots/"
        self.key_frame = ""
        self.saved_frames = []

    async def _encode_image(self, img):
        """Encode image as base64"""
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        return base64_image

    async def _save_clip(
        self, clip_data=None, clip_path=None, image_data=None, image_path=None
    ):
        _LOGGER.debug(f"Saving clip to {clip_path} and image to {image_path}")
        # Ensure dir exists
        await self.hass.loop.run_in_executor(
            None, partial(os.makedirs, self.snapshots_path, exist_ok=True)
        )

        def _run_save_clips(clip_data, clip_path, image_data, image_path):
            _LOGGER.info(f"[save_clip] clip: {clip_path}, image: {image_path}")
            if image_data:
                with open(image_path, "wb") as f:
                    if type(image_data) == bytes:
                        f.write(image_data)
                    else:
                        f.write(base64.b64decode(image_data))
            elif clip_data:
                with open(clip_path, "wb") as f:
                    f.write(clip_data)

        await self.hass.loop.run_in_executor(
            None, _run_save_clips, clip_data, clip_path, image_data, image_path
        )

    def _convert_to_rgb(self, img):
        if img.mode == "RGBA" or img.format == "GIF":
            img = img.convert("RGB")
        return img

    async def _expose_image(self, frame_name, image_data, uid, frame_path=None):
        snapshots_dir = self.snapshots_path.rstrip("/\\")
        await self.hass.loop.run_in_executor(
            None,
            partial(os.makedirs, snapshots_dir, exist_ok=True),
        )
        filename = f"{snapshots_dir}/{uid}-{frame_name}.jpg"
        if image_data is None and frame_path is not None:
            # open image in hass.loop
            with await self.hass.loop.run_in_executor(
                None, Image.open, frame_path
            ) as image:
                await self.hass.loop.run_in_executor(None, image.load)
                image_data = await self._encode_image(image)
        await self._save_clip(image_data=image_data, image_path=filename)
        if filename not in self.saved_frames:
            self.saved_frames.append(filename)
        if not self.key_frame:
            self.key_frame = filename
        return filename

    def _similarity_score(self, previous_frame, current_frame_gray):
        """
        SSIM by Z. Wang: https://ece.uwaterloo.ca/~z70wang/research/ssim/
        Paper:  Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
        "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
        """
        K1 = 0.005
        K2 = 0.015
        L = 255

        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        previous_frame_np = np.array(previous_frame)
        current_frame_np = np.array(current_frame_gray)

        # Ensure both frames have same dimensions
        if previous_frame_np.shape != current_frame_np.shape:
            min_shape = np.minimum(previous_frame_np.shape, current_frame_np.shape)
            previous_frame_np = previous_frame_np[: min_shape[0], : min_shape[1]]
            current_frame_np = current_frame_np[: min_shape[0], : min_shape[1]]

        # Calculate mean (mu)
        mu1 = np.mean(previous_frame_np, dtype=np.float64)
        mu2 = np.mean(current_frame_np, dtype=np.float64)

        # Calculate variance (sigma^2) and covariance (sigma12)
        sigma1_sq = np.var(previous_frame_np, dtype=np.float64, mean=mu1)
        sigma2_sq = np.var(current_frame_np, dtype=np.float64, mean=mu2)
        sigma12 = np.cov(
            previous_frame_np.flatten(), current_frame_np.flatten(), dtype=np.float64
        )[0, 1]

        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim

    async def _select_keyframe_index(
        self, reference_frame_bytes, candidate_frames_bytes
    ):
        """
        Pick the index of the frame most different from the reference frame.
        """
        # Decode reference to grayscale
        ref_img = Image.open(io.BytesIO(reference_frame_bytes))
        try:
            await self.hass.loop.run_in_executor(None, ref_img.load)
            ref_gray = np.array(ref_img.convert("L"))
        finally:
            ref_img.close()

        best_idx = 0
        best_score = float("inf")  # minimize SSIM
        for idx, frame_bytes in enumerate(candidate_frames_bytes):
            img = Image.open(io.BytesIO(frame_bytes))
            try:
                await self.hass.loop.run_in_executor(None, img.load)
                curr_gray = np.array(img.convert("L"))
            finally:
                img.close()
            score = self._similarity_score(ref_gray, curr_gray)
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    async def resize_image(
        self, target_width, image_path=None, image_data=None, img=None
    ):
        """Resize image to target_width"""
        base64_image = None

        if image_path:
            # Open the image file
            img = await self.hass.loop.run_in_executor(None, Image.open, image_path)
            with img:
                await self.hass.loop.run_in_executor(None, img.load)
                # Check if the image is a GIF and convert if necessary
                img = self._convert_to_rgb(img)
                # calculate new height based on aspect ratio
                width, height = img.size
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)

                # Resize the image only if it's larger than the target size
                if width > target_width or height > target_height:
                    img = img.resize((target_width, target_height))

                # Encode the image to base64
                base64_image = await self._encode_image(img)

        elif image_data:
            # Convert the image to base64
            img_byte_arr = io.BytesIO()
            img_byte_arr.write(image_data)
            img = await self.hass.loop.run_in_executor(None, Image.open, img_byte_arr)
            with img:
                await self.hass.loop.run_in_executor(None, img.load)
                img = self._convert_to_rgb(img)
                # calculate new height based on aspect ratio
                width, height = img.size
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)

                if width > target_width or height > target_height:
                    img = img.resize((target_width, target_height))

                base64_image = await self._encode_image(img)
        elif img:
            with img:
                img = self._convert_to_rgb(img)
                # calculate new height based on aspect ratio
                width, height = img.size
                aspect_ratio = width / height
                target_height = int(target_width / aspect_ratio)

                if width > target_width or height > target_height:
                    img = img.resize((target_width, target_height))

                base64_image = await self._encode_image(img)

        if base64_image is None:
            raise ServiceValidationError("No image data provided for resize_image")

        return base64_image

    async def _fetch(
        self, url, target_file=None, max_retries=2, retry_delay=1, entity_name=None
    ):
        """Fetch image from url and return image data"""
        retries = 0
        entity_prefix = f"Camera {entity_name}: " if entity_name else ""
        while retries < max_retries:
            _LOGGER.info(f"Fetching {url} (attempt {retries + 1}/{max_retries})")
            try:
                async with self.session.get(url) as response:
                    if not response.ok:
                        _LOGGER.warning(
                            f"{entity_prefix}Couldn't fetch frame (status code: {response.status})"
                        )
                        retries += 1
                        await asyncio.sleep(retry_delay)
                        continue
                    # Just read file into buffer
                    if target_file is None:
                        data = await response.read()
                        return data
                    else:  # Save response into file in stream fashion to avoid memory leaks
                        _LOGGER.debug(f"writing response into file {target_file}")
                        written = 0
                        chunks = 0
                        async with async_open(target_file, "wb") as output:
                            async for data in response.content.iter_any():
                                await output.write(data)
                                written += len(data)
                                chunks += 1
                        _LOGGER.debug(
                            f"wrote {written} bytes ({chunks} chunks) into {target_file}"
                        )

                        return None
            except Exception as e:
                _LOGGER.error(f"{entity_prefix}Fetch failed: {e}")
                retries += 1
                await asyncio.sleep(retry_delay)
        _LOGGER.warning(
            f"{entity_prefix}Failed to fetch {url} after {max_retries} retries"
        )

    async def record(
        self,
        image_entities,
        duration,
        max_frames,
        target_width,
        include_filename,
        expose_images,
        lookback=0,
    ):
        """Wrapper for client.add_frame with integrated recorder

        Args:
            image_entities (list[string]): List of camera entities to record
            duration (float): Duration in seconds to record
            target_width (int): Target width for the images in pixels
            lookback (int): Pre-event lookback duration in seconds
        """

        if isinstance(image_entities, str):
            image_entities = [image_entities]

        if lookback and lookback > 0:
            return await self._record_with_lookback(
                image_entities=image_entities,
                duration=duration,
                lookback=lookback,
                max_frames=max_frames,
                target_width=target_width,
                include_filename=include_filename,
                expose_images=expose_images,
            )

        if duration is None or duration < 3:
            interval = 1
        elif duration < 10:
            interval = 2
        elif duration < 30:
            interval = 3
        elif duration < 60:
            interval = 5
        else:
            interval = 5
        camera_frames = {}
        first_frames = {}
        # Track successful image entities (cameras that successfully captured frames)
        successful_image_entities = set()

        # Record on a separate thread for each camera
        async def record_camera(image_entity, camera_number):
            start = time.time()
            frame_counter = 0
            frames = {}
            previous_frame = None
            iteration_time = 0

            base_url = get_url(self.hass)

            while time.time() - start < duration + iteration_time:
                fetch_start_time = time.time()
                entity_state = self.hass.states.get(image_entity)

                # Check if entity exists
                if entity_state is None:
                    _LOGGER.error(f"Camera {image_entity} does not exist")
                    await asyncio.sleep(interval)
                    continue

                entity_picture = entity_state.attributes.get("entity_picture")

                # Skip if camera is offline or entity_picture unavailable
                if not entity_picture:
                    _LOGGER.warning(
                        f"Camera {image_entity} is offline or does not have entity_picture attribute"
                    )
                    await asyncio.sleep(interval)
                    continue

                frame_url = base_url + entity_picture
                frame_data = await self._fetch(frame_url, entity_name=image_entity)

                # Skip frame if fetch failed
                if not frame_data:
                    await asyncio.sleep(interval)
                    continue

                fetch_duration = time.time() - fetch_start_time
                _LOGGER.info(f"Fetched {image_entity} in {fetch_duration:.2f} seconds")

                preprocessing_start_time = time.time()

                with await self.hass.loop.run_in_executor(
                    None, Image.open, io.BytesIO(frame_data)
                ) as img:
                    current_frame_gray = np.array(img.convert("L"))

                    if previous_frame is not None:
                        score = self._similarity_score(
                            previous_frame, current_frame_gray
                        )
                        # Encode the image back to bytes
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG")
                        frame_data = buffer.getvalue()

                        # Use either entity name or assign number to each camera
                        if include_filename:
                            parts = [
                                image_entity.replace("camera.", ""),
                                "frame",
                                str(frame_counter),
                            ]
                        else:
                            parts = [
                                f"camera{camera_number}",
                                "frame",
                                str(frame_counter),
                            ]
                        frame_label = "-".join(parts)
                        frames.update(
                            {
                                frame_label: {
                                    "frame_data": frame_data,
                                    "ssim_score": score,
                                    "camera_number": camera_number,
                                    "frame_index": frame_counter,
                                }
                            }
                        )

                        frame_counter += 1
                        previous_frame = current_frame_gray
                    else:
                        # Current frame is first frame
                        previous_frame = current_frame_gray
                        # Normalize to JPEG
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG")
                        first_bytes = buffer.getvalue()
                        if include_filename:
                            parts = [
                                image_entity.replace("camera.", ""),
                                "frame",
                                str(frame_counter),
                            ]
                        else:
                            parts = [
                                f"camera{camera_number}",
                                "frame",
                                str(frame_counter),
                            ]
                        frame_label = "-".join(parts)
                        first_frames[image_entity] = (frame_label, first_bytes)
                        # Mark this camera as successful
                        successful_image_entities.add(image_entity)
                        frame_counter += 1

                preprocessing_duration = time.time() - preprocessing_start_time
                _LOGGER.info(
                    f"Preprocessing took: {preprocessing_duration:.2f} seconds"
                )

                adjusted_interval = max(
                    0, interval - fetch_duration - preprocessing_duration
                )

                if iteration_time == 0:
                    iteration_time = time.time() - start
                    _LOGGER.info(
                        f"First iteration took: {iteration_time:.2f} seconds, interval adjusted to: {adjusted_interval}"
                    )

                await asyncio.sleep(adjusted_interval)

            camera_frames.update({image_entity: frames})

        camera_names = ", ".join(
            entity.replace("camera.", "") for entity in image_entities
        )
        _LOGGER.info(f"Recording {camera_names} for {duration} seconds")

        # start threads for each camera
        await asyncio.gather(
            *(
                record_camera(image_entity, image_entities.index(image_entity))
                for image_entity in image_entities
            )
        )

        # Check if any cameras successfully captured frames
        if len(successful_image_entities) == 0:
            raise ServiceValidationError(
                "No cameras available - all cameras offline or unavailable"
            )

        # Extract frames and their SSIM scores
        frames_with_scores = []
        for frame in camera_frames:
            for frame_name, frame_data in camera_frames[frame].items():
                frames_with_scores.append(
                    (
                        frame_name,
                        frame_data["frame_data"],
                        frame_data["ssim_score"],
                        frame_data["camera_number"],
                        frame_data["frame_index"],
                    )
                )

        # Sort frames by SSIM score
        frames_with_scores.sort(key=lambda x: x[2])

        # Frame selection: prepend first frames, then best-scored (respect max_frames)
        selected_frames = []
        remaining = max(0, max_frames)

        # Prepend first frames in the order of requested entities
        for entity in image_entities:
            if remaining <= 0:
                break
            if entity in first_frames:
                label, data = first_frames[entity]
                selected_frames.append((label, data, None))
                remaining -= 1

        # Fill remaining slots with best scored frames, then restore stable capture order
        best_rest = frames_with_scores[:remaining]
        best_rest.sort(key=lambda x: (x[4], x[3]))
        for name, data, score, _, _ in best_rest:
            selected_frames.append((name, data, score))

        # Add selected frames to client
        if selected_frames:
            # Choose keyframe among the selected frames using the last as reference
            reference_bytes = selected_frames[-1][1]
            candidate_bytes = [data for _, data, _ in selected_frames]
            key_idx = await self._select_keyframe_index(
                reference_bytes, candidate_bytes
            )

            # Add all frames (resized) and expose only the chosen keyframe
            resized_base64 = []
            for frame_name, frame_data, _ in selected_frames:
                resized_image = await self.resize_image(
                    target_width=target_width, image_data=frame_data
                )
                resized_base64.append(resized_image)
                self.client.add_frame(base64_image=resized_image, filename=frame_name)

            if expose_images:
                uid = str(uuid.uuid4())[:8]
                start_saved_len = len(self.saved_frames)
                for idx, (frame_name, _, _) in enumerate(selected_frames, start=1):
                    await self._expose_image(
                        frame_name=f"{frame_name}",
                        image_data=resized_base64[idx - 1],
                        uid=uid,
                    )
                if start_saved_len + key_idx < len(self.saved_frames):
                    self.key_frame = self.saved_frames[start_saved_len + key_idx]

    async def add_images(
        self, image_entities, image_paths, target_width, include_filename, expose_images
    ):
        """Wrapper for client.add_frame for images"""
        base_url = get_url(self.hass)
        # Track successful image entities (cameras that successfully provided frames)
        successful_image_entities = 0

        if image_entities:
            for image_entity in image_entities:
                try:
                    entity_state = self.hass.states.get(image_entity)

                    # Check if entity exists
                    if entity_state is None:
                        _LOGGER.error(f"Camera {image_entity} does not exist")
                        continue

                    entity_picture = entity_state.attributes.get("entity_picture")

                    # Skip if camera is offline or entity_picture unavailable
                    if not entity_picture:
                        _LOGGER.warning(
                            f"Camera {image_entity} is offline or does not have entity_picture attribute"
                        )
                        continue

                    image_url = base_url + entity_picture
                    image_data = await self._fetch(image_url, entity_name=image_entity)

                    # Skip frame if fetch failed
                    if not image_data:
                        _LOGGER.warning(f"Camera {image_entity}: Failed to fetch image")
                        continue

                    # If entity snapshot requested, use entity name as 'filename'
                    resized_image = await self.resize_image(
                        target_width=target_width, image_data=image_data
                    )
                    self.client.add_frame(
                        base64_image=resized_image,
                        filename=(
                            entity_state.attributes.get("friendly_name")
                            if include_filename
                            else ""
                        ),
                    )

                    if expose_images:
                        await self._expose_image(
                            frame_name="0",
                            image_data=resized_image,
                            uid=str(uuid.uuid4())[:8],
                        )

                    successful_image_entities += 1

                except AttributeError as e:
                    _LOGGER.error(
                        f"Camera {image_entity}: AttributeError accessing entity attributes: {e}"
                    )
                    raise ServiceValidationError(
                        f"Error accessing camera entity {image_entity}: {e}"
                    )

            # Check if any cameras were successful
            if successful_image_entities == 0:
                raise ServiceValidationError(
                    "No cameras available - all cameras offline or unavailable"
                )
        if image_paths:
            for image_path in image_paths:
                try:
                    image_path = image_path.strip()

                    if not os.path.exists(image_path):
                        raise ServiceValidationError(
                            f"File {image_path} does not exist"
                        )

                    filename = ""

                    if include_filename:
                        filename = image_path.split("/")[-1].split(".")[-2]

                    image_data = await self.resize_image(
                        target_width=target_width, image_path=image_path
                    )

                    self.client.add_frame(base64_image=image_data, filename=filename)

                    if expose_images:
                        await self._expose_image(
                            frame_name="0",
                            image_data=image_data,
                            uid=str(uuid.uuid4())[:8],
                        )
                except Exception as e:
                    raise ServiceValidationError(f"Error: {e}")
        return self.client

    async def add_video(
        self,
        video_path,
        base_url,
        max_frames=10,
        target_width=640,
        include_filename=False,
        expose_images=False,
        override_filename="",
    ):
        try:
            current_event_id = str(uuid.uuid4())
            video_path = video_path.strip()

            # Resolve media source (media-source://...)
            if is_media_source_id(video_path):
                _LOGGER.debug(f"Resolving media source id: {video_path}")
                video_path = async_process_play_media_url(self.hass, video_path)
                _LOGGER.debug(f"media url = {video_path}")

            # Sign local API URLs unless already signed
            if video_path.startswith("/api"):
                if "authSig" not in video_path:
                    # Add authorization signature with 5 minute expiration
                    _LOGGER.debug(f"Signing {video_path}")
                    video_path = async_sign_path(
                        self.hass, video_path, timedelta(minutes=5)
                    )
                    _LOGGER.debug(f"signed_path = {video_path}")
                else:
                    # Already signed, just use it
                    _LOGGER.debug(f"Already signed {video_path}")

                video_path = base_url + video_path

            ffmpeg_tail = [
                "-vf",  # video filter
                "select=eq(pict_type\\,I)",  # select only I-frames
                "-vsync",
                "0",  # disable v-sync to avoid frame duplication
                "-q:v",  # quality level for JPEG (lower is better)
                "5",  # 5 medium quality
                "-f",
                "image2pipe",  # output to pipe
                "-vcodec",
                "mjpeg",  # encode as mjpeg
                "-",
            ]
            ffmpeg_stderr = None
            temp_file_path = None

            # If file is served over http(s)
            if video_path.startswith("http://") or video_path.startswith("https://"):
                # Download to a seekable temp file so ffmpeg can parse MP4 (moov at end)
                suffix = os.path.splitext(urlparse(video_path).path)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    temp_file_path = tmp.name
                await self._fetch(video_path, target_file=temp_file_path)
                if (
                    not os.path.exists(temp_file_path)
                    or os.path.getsize(temp_file_path) == 0
                ):
                    raise ServiceValidationError(
                        f"Failed to fetch video from {video_path}"
                    )

                ffmpeg_cmd = [
                    "ffmpeg",
                    "-hide_banner",  # cleaner logs
                    "-loglevel",
                    "error",
                    # input network robustness if ever used with URLs directly
                    "-an",
                    "-sn",
                    "-dn",
                    "-i",
                    temp_file_path,
                    *ffmpeg_tail,
                ]

                output = asyncio.subprocess.PIPE
                error_output = asyncio.subprocess.DEVNULL
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    error_output = asyncio.subprocess.PIPE
                ffmpeg_stderr = None

                ffmpeg_start = time.monotonic_ns()
                ffmpeg_timeout = 300  # seconds

                _LOGGER.debug(
                    f"Running FFMPEG to create keyframes: {' '.join(ffmpeg_cmd)}"
                )
                ffmpeg_process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=output,
                    stderr=error_output,
                )

            else:
                # Local file
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-hwaccel",
                    "auto",
                    "-an",
                    "-sn",
                    "-dn",
                    "-i",
                    video_path,
                    *ffmpeg_tail,
                ]

                output = asyncio.subprocess.PIPE
                error_output = asyncio.subprocess.DEVNULL
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    error_output = None

                ffmpeg_start = time.monotonic_ns()
                ffmpeg_timeout = 300  # seconds

                _LOGGER.debug(
                    f"Running FFMPEG to create keyframes: {' '.join(ffmpeg_cmd)}"
                )
                ffmpeg_process = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd, stdout=output, stderr=error_output
                )
                _LOGGER.info(
                    f"Started ffmpeg pid={ffmpeg_process.pid} (stdin={'inherit' if video_path else 'none'}, "
                    f"stdout={'pipe' if ffmpeg_process.stdout else 'none'}, stderr={'inherited' if error_output is None else 'devnull'})"
                )

            previous_frame = None
            candidate_heap = []
            frame_counter = 0
            jpeg_buffer = b""
            first_frame = None

            max_candidates = max(
                max_frames or 1,
                min((max_frames or 5) * 10, MAX_KEYFRAME_CANDIDATES),
            )

            def find_jpeg_frames(data):
                frames = []
                start = 0
                while True:
                    soi = data.find(b"\xff\xd8", start)
                    eoi = data.find(b"\xff\xd9", soi)
                    if soi == -1 or eoi == -1:
                        break
                    frames.append(data[soi : eoi + 2])
                    start = eoi + 2
                return frames, data[start:]

            try:
                per_read_timeout = 30  # seconds
                # Ensure stdout is readable
                if ffmpeg_process.stdout is None:
                    _LOGGER.error("ffmpeg stdout is not a PIPE; cannot read frames")
                    await ffmpeg_process.wait()
                    raise ServiceValidationError("ffmpeg stdout not available")
                # Read until ffmpeg closes stdout (evaluate full clip / drain to EOF)
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            ffmpeg_process.stdout.read(4096), timeout=per_read_timeout
                        )
                    except asyncio.TimeoutError:
                        _LOGGER.warning(
                            "Timeout while waiting for ffmpeg stdout read; terminating read loop"
                        )
                        break

                    if not chunk:
                        _LOGGER.debug("ffmpeg stdout closed or returned no data")
                        break

                    jpeg_buffer += chunk
                    found_frames, jpeg_buffer = find_jpeg_frames(jpeg_buffer)

                    for jpeg_data in found_frames:
                        try:
                            img = Image.open(io.BytesIO(jpeg_data))
                            await self.hass.loop.run_in_executor(None, img.load)
                            if img.mode == "RGBA":
                                img = img.convert("RGB")
                            current_frame_gray = np.array(img.convert("L"))
                            if previous_frame is not None:
                                score = self._similarity_score(
                                    previous_frame, current_frame_gray
                                )
                                candidate_item = (jpeg_data, score, frame_counter)
                                heap_entry = (-score, -frame_counter, candidate_item)
                                if len(candidate_heap) < max_candidates:
                                    heapq.heappush(candidate_heap, heap_entry)
                                else:
                                    if score < -candidate_heap[0][0]:
                                        heapq.heappushpop(candidate_heap, heap_entry)
                                _LOGGER.debug(
                                    f"Evaluated frame {frame_counter} (score={score:.6f}, bytes={len(jpeg_data)})"
                                )
                            else:
                                # First frame, always include
                                first_frame = (jpeg_data, frame_counter)
                                _LOGGER.debug(
                                    f"Captured first frame {frame_counter} (bytes={len(jpeg_data)})"
                                )

                            previous_frame = current_frame_gray
                            frame_counter += 1
                        except UnidentifiedImageError:
                            _LOGGER.error(
                                f"Cannot identify image from ffmpeg pipe at frame {frame_counter}"
                            )
                            continue
                await ffmpeg_process.wait()

                if (
                    error_output == asyncio.subprocess.PIPE
                    and ffmpeg_process.stderr is not None
                ):
                    try:
                        ffmpeg_stderr = await ffmpeg_process.stderr.read()
                    except Exception:
                        ffmpeg_stderr = None

            except asyncio.TimeoutError:
                _LOGGER.info(
                    f"FFmpeg failed to process video within {ffmpeg_timeout} seconds"
                )
                if ffmpeg_process.returncode is not None:
                    ffmpeg_process.terminate()

            _LOGGER.debug(
                f"FFmpeg process finished with return code {ffmpeg_process.returncode}"
            )

            # Cleanup temp file (if any)
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

            if ffmpeg_process.returncode != 0:
                msg = f"FFmpeg failed with return code {ffmpeg_process.returncode}"
                if ffmpeg_stderr:
                    try:
                        msg += f": {ffmpeg_stderr.decode(errors='ignore')}"
                    except Exception:
                        pass
                raise ServiceValidationError(msg)

            ffmpeg_time = time.monotonic_ns() - ffmpeg_start
            _LOGGER.debug(f"FFmpeg took {ffmpeg_time / 1_000_000:.2f} ms")

            frames = [item[2] for item in candidate_heap]
            if len(frames) == 0 and first_frame is None:
                raise ServiceValidationError("No frames extracted from video.")

            # Log all frames with scores
            for fdata, fscore, findex in frames:
                _LOGGER.debug(f"Candidate frame {findex} with SSIM score {fscore:.6f}")

            # Sort scored frames by SSIM
            frames.sort(key=lambda x: x[1])

            # Frame selection: prepend first frame, then best-scored (respect max_frames)
            selected_frames = []
            remaining = max(0, max_frames)

            if first_frame is not None and remaining > 0:
                first_data, first_idx = first_frame
                selected_frames.append((first_data, None, first_idx))
                remaining -= 1

            best_rest = frames[:remaining]
            # Keep chronological order for the rest
            best_rest.sort(key=lambda x: x[2])
            selected_frames.extend(best_rest)

            # Add frames to client
            resized_base64 = []
            for idx, (frame_data, _, _) in enumerate(selected_frames, start=1):
                resized_image = await self.resize_image(
                    target_width=target_width, image_data=frame_data
                )
                resized_base64.append(resized_image)
                if override_filename:
                    filename = f"{override_filename} (frame {idx})"
                elif include_filename:
                    filename = f"{os.path.splitext(os.path.basename(video_path))[0]} (frame {idx})"
                else:
                    filename = f"Video frame {idx}"
                self.client.add_frame(
                    base64_image=resized_image,
                    filename=filename,
                )

            if expose_images and selected_frames:
                uid = str(uuid.uuid4())[:8]
                reference_bytes = selected_frames[0][0]
                candidate_bytes = [fd for (fd, _, _) in selected_frames]
                key_idx = await self._select_keyframe_index(
                    reference_bytes, candidate_bytes
                )
                raw_base_name = (
                    override_filename
                    if override_filename
                    else os.path.splitext(os.path.basename(video_path))[0]
                )
                safe_base_name = _sanitize_filename(raw_base_name)
                exposed_video_frames = []
                for idx, (frame_data, _, orig_idx) in enumerate(selected_frames, start=1):
                    frame_idx_label = (orig_idx if orig_idx is not None else idx - 1) + 1
                    written_file = await self._expose_image(
                        frame_name=f"{safe_base_name}-frame_{frame_idx_label}",
                        image_data=resized_base64[idx - 1],
                        uid=uid,
                    )
                    if written_file:
                        exposed_video_frames.append(written_file)
                if exposed_video_frames and 0 <= key_idx < len(exposed_video_frames):
                    self.key_frame = exposed_video_frames[key_idx]
        except Exception as e:
            raise ServiceValidationError(f"Error processing video {video_path}: {e}")

    async def add_videos(
        self,
        video_paths,
        event_ids,
        max_frames,
        target_width,
        include_filename,
        expose_images,
    ):
        """Wrapper for client.add_frame for videos"""

        if not video_paths:
            video_paths = []
        else:
            video_paths = list(video_paths)

        base_url = get_url(self.hass)

        if event_ids:
            for event_id in event_ids:
                url = "/api/frigate/notifications/" + event_id + "/clip.mp4"
                # append to video_paths
                video_paths.append(url)

        _LOGGER.debug(f"Processing videos: {video_paths}")

        # Process videos sequentially in input order for deterministic aggregation
        for video_path in video_paths:
            await self.add_video(
                video_path=video_path,
                base_url=base_url,
                max_frames=max_frames,
                target_width=target_width,
                include_filename=include_filename,
                expose_images=expose_images,
            )

        return self.client

    async def _safe_cleanup_recorder_provider(self, image_entity, temp_filename):
        """Safely release only matching owned RecorderOutput provider from stream.

        Matches current native outputs()['recorder'].video_path EXACTLY to temp_filename.
        Leaves HLS and borrowed providers untouched, never calls stream.stop().
        """
        try:
            stream = None
            if DOMAIN in self.hass.data and "active_streams" in self.hass.data[DOMAIN]:
                stream = self.hass.data[DOMAIN]["active_streams"].get(image_entity)
            if stream is None and hasattr(self.hass, "data") and "camera" in self.hass.data:
                camera_comp = self.hass.data.get("camera")
                if hasattr(camera_comp, "get_entity"):
                    cam_ent = camera_comp.get_entity(image_entity)
                    if cam_ent and hasattr(cam_ent, "stream"):
                        stream = cam_ent.stream

            if stream is not None and hasattr(stream, "remove_provider"):
                outputs = stream.outputs() if callable(getattr(stream, "outputs", None)) else getattr(stream, "outputs", {})
                recorder = None
                if isinstance(outputs, dict):
                    recorder = outputs.get("recorder")
                elif isinstance(outputs, (list, tuple)):
                    for p in outputs:
                        if getattr(p, "name", None) == "recorder" or getattr(p, "video_path", None) == temp_filename:
                            recorder = p
                            break

                if recorder is not None and getattr(recorder, "video_path", None) == temp_filename:
                    _LOGGER.debug(f"Safely removing matching owned recorder provider for {image_entity}")
                    res = stream.remove_provider(recorder)
                    if asyncio.iscoroutine(res):
                        await res
        except Exception as e:
            _LOGGER.debug(f"Stream recorder provider cleanup skipped for {image_entity}: {e}")

    async def _record_with_lookback(
        self,
        image_entities,
        duration,
        lookback,
        max_frames,
        target_width,
        include_filename,
        expose_images,
    ):
        """Record video clip using Home Assistant camera.record with lookback and extract frames.

        Core camera.record semantics and safe provider release disclosure:
        Home Assistant Core's camera.record service creates a RecorderOutput provider on the stream.
        When cancelling or on sibling error, our owned RecorderOutput can be released safely via
        await stream.remove_provider(provider) ONLY if provider.video_path EXACTLY matches our unique
        temp_filename, leaving HLS and borrowed providers untouched.
        Cancelling and joining the Python camera.record service task gives an owned task a bounded 2s
        opportunity to finish, but cannot guarantee halting already-running native executor/thread jobs
        inside Core or underlying libraries. If an executor job outlives cancellation, late filesystem
        writes may occur before the OS file handle is closed.
        Temporary files (including Core's .tmp files) are cleaned up on a best-effort basis without
        calling stream.stop() or disrupting active HLS viewers.
        If native stream provider APIs are unavailable, failure is raised explicitly.
        """
        if isinstance(image_entities, str):
            image_entities = [image_entities]

        snapshots_dir = self.snapshots_path.rstrip("/\\")
        await self.hass.loop.run_in_executor(
            None, partial(os.makedirs, snapshots_dir, exist_ok=True)
        )
        base_url = get_url(self.hass)

        temp_files_to_cleanup = []

        async def record_single_camera_clip(image_entity):
            entity_state = self.hass.states.get(image_entity)
            if entity_state is None:
                _LOGGER.error(f"Camera {image_entity} does not exist")
                raise ServiceValidationError(f"Camera {image_entity} does not exist")

            friendly_name = (
                entity_state.attributes.get("friendly_name")
                or image_entity.replace("camera.", "")
            )
            uid = str(uuid.uuid4())[:8]
            temp_filename = f"{snapshots_dir}/temp_lookback_{uid}.mp4"
            temp_files_to_cleanup.append(temp_filename)

            _LOGGER.info(
                f"Recording camera {image_entity} with lookback={lookback}s, duration={duration}s to {temp_filename}"
            )
            service_task = asyncio.create_task(
                self.hass.services.async_call(
                    "camera",
                    "record",
                    {
                        "entity_id": image_entity,
                        "filename": temp_filename,
                        "duration": duration,
                        "lookback": lookback,
                    },
                    blocking=True,
                )
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(service_task),
                    timeout=duration + lookback + 20,
                )
            except asyncio.CancelledError:
                # Orchestration worker was cancelled (e.g. caller cancelled or sibling failed).
                # Shield owned service task and await natural completion with bounded deadline
                # so native call finishes/closes files before cleanup.
                try:
                    await asyncio.wait_for(asyncio.shield(service_task), timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    if not service_task.done():
                        service_task.cancel()
                        try:
                            await service_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    await self._safe_cleanup_recorder_provider(image_entity, temp_filename)
                raise
            except asyncio.TimeoutError:
                if not service_task.done():
                    service_task.cancel()
                    try:
                        await service_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await self._safe_cleanup_recorder_provider(image_entity, temp_filename)
                _LOGGER.error(
                    f"Recording timed out for camera {image_entity} after {duration + lookback + 20}s. The stream source may be unreachable."
                )
                raise ServiceValidationError(
                    f"Recording timed out for {image_entity}. Please check that the camera RTSP stream is online and reachable."
                )
            except Exception as e:
                if not service_task.done():
                    service_task.cancel()
                    try:
                        await service_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await self._safe_cleanup_recorder_provider(image_entity, temp_filename)
                _LOGGER.error(
                    f"Error recording camera {image_entity} with lookback: {e}"
                )
                if isinstance(e, ServiceValidationError):
                    raise
                raise ServiceValidationError(
                    f"Error recording camera {image_entity} with lookback: {e}"
                )

            file_exists = await self.hass.loop.run_in_executor(
                None, os.path.exists, temp_filename
            )
            file_size = (
                await self.hass.loop.run_in_executor(
                    None, os.path.getsize, temp_filename
                )
                if file_exists
                else 0
            )
            _LOGGER.info(
                f"Camera {image_entity} recorded: exists={file_exists}, size={file_size} bytes"
            )
            if not file_exists or file_size == 0:
                _LOGGER.error(
                    f"Failed to record stream for {image_entity} with lookback={lookback}s. The file is empty or does not exist."
                )
                raise ServiceValidationError(
                    f"Failed to record stream for {image_entity}. The camera stream may not be active or does not support recording."
                )

            return {
                "entity": image_entity,
                "temp_filename": temp_filename,
                "friendly_name": friendly_name,
            }

        tasks = []
        try:
            tasks = [
                asyncio.create_task(record_single_camera_clip(entity))
                for entity in image_entities
            ]

            if tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_EXCEPTION
                )

                exc = None
                for t in done:
                    if t.exception() is not None:
                        exc = t.exception()
                        break

                if exc is not None or pending:
                    if exc is not None:
                        # Cancel pending sibling tasks immediately on first error
                        for t in pending:
                            t.cancel()
                        # Join cancelled sibling tasks
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        if isinstance(exc, ServiceValidationError):
                            raise exc
                        raise ServiceValidationError(
                            f"Error recording camera stream with lookback: {exc}"
                        )
                    else:
                        await asyncio.gather(*pending)

            # Retrieve recorded clip metadata in deterministic input order
            recorded_clips = [t.result() for t in tasks]

            # Process clips sequentially in input order
            for clip in recorded_clips:
                temp_filename = clip["temp_filename"]
                friendly_name = clip["friendly_name"]
                override_name = friendly_name if include_filename else ""
                try:
                    await self.add_video(
                        video_path=temp_filename,
                        base_url=base_url,
                        max_frames=max_frames,
                        target_width=target_width,
                        include_filename=include_filename,
                        expose_images=expose_images,
                        override_filename=override_name,
                    )
                except Exception as e:
                    if isinstance(e, ServiceValidationError):
                        raise
                    raise ServiceValidationError(
                        f"Error processing clip for {clip['entity']}: {e}"
                    )

            if len(recorded_clips) == 0:
                raise ServiceValidationError(
                    "No camera recordings could be captured with lookback. Ensure cameras have stream support enabled."
                )

            return self.client

        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            raise

        finally:
            def _cleanup():
                for fpath in temp_files_to_cleanup:
                    for target in (fpath, f"{fpath}.tmp"):
                        if os.path.exists(target):
                            try:
                                os.remove(target)
                            except Exception as e:
                                _LOGGER.debug(
                                    f"Failed to remove temp file {target}: {e}"
                                )

            await self.hass.loop.run_in_executor(None, _cleanup)

    async def add_streams(
        self,
        image_entities,
        duration,
        max_frames,
        target_width,
        include_filename,
        expose_images,
        lookback=0,
    ):
        if image_entities:
            await self.record(
                image_entities=image_entities,
                duration=duration,
                max_frames=max_frames,
                target_width=target_width,
                include_filename=include_filename,
                expose_images=expose_images,
                lookback=lookback,
            )
        return self.client

    async def add_visual_data(
        self, image_entities, image_paths, target_width, include_filename, expose_images
    ):
        """Wrapper for add_images for visual data"""
        await self.add_images(
            image_entities=image_entities,
            image_paths=image_paths,
            target_width=target_width,
            include_filename=include_filename,
            expose_images=expose_images,
        )
        return self.client

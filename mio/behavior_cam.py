"""
Behavior camera capture using multiprocessing for USB cameras.
"""

import multiprocessing
import os
import queue
import time
from pathlib import Path
from typing import Optional, Union

import cv2

from mio import init_logger
from mio.devices.usbcam import convert_frame_for_codec, determine_pix_fmt, open_camera
from mio.io import BufferedCSVWriter, VideoWriter
from mio.models.usbcam import USBCameraRecordingConfig
from mio.types import ConfigSource
from mio.utils import exact_iter

FPS_LOG_INTERVAL_SECONDS = 10.0
FRAME_QUEUE_MAXSIZE = 100
CSV_BUFFER_SIZE = 100
QUEUE_TIMEOUT_SECONDS = 1.0


class BehaviorCam:
    """
    Behavior camera capture class using multiprocessing.

    Separates camera interface from capture/writing logic using multiprocessing.
    Similar architecture to :class:`.StreamDaq`.
    """

    def __init__(
        self,
        recording_config: Union[USBCameraRecordingConfig, ConfigSource],
        camera_index: int,
    ) -> None:
        """
        Initialize behavior camera capture.

        Args:
            recording_config: Configuration object, config ID, or path to config file
            camera_index: Index of the camera to use
        """
        self.logger = init_logger("behaviorCam")
        self.config = USBCameraRecordingConfig.from_any(recording_config)
        self.camera_index = camera_index
        self.terminate: multiprocessing.Event = multiprocessing.Event()

    def _camera_recv(
        self,
        frame_queue: multiprocessing.Queue,
    ) -> None:
        """
        Read frames from camera and put them in the queue.

        This runs in a separate process to decouple camera I/O from writing.

        Args:
            frame_queue: Queue to put frames into
        """
        locallogs = init_logger("behaviorCam.camera_recv")

        try:
            cap = open_camera(
                camera_index=self.camera_index,
                frame_width=self.config.frame_width,
                frame_height=self.config.frame_height,
                fps=self.config.fps,
            )
        except RuntimeError:
            frame_queue.put(None)
            raise

        locallogs.info("Camera opened, starting frame capture")

        try:
            while not self.terminate.is_set():
                ret, frame = cap.read()
                if ret:
                    # Get timestamp for this frame (float unix time in seconds)
                    unix_time = time.time()
                    try:
                        frame_queue.put(
                            (frame, unix_time),
                            block=True,
                            timeout=QUEUE_TIMEOUT_SECONDS,
                        )
                    except queue.Full:
                        locallogs.warning("Frame queue full, skipping frame")
                else:
                    locallogs.warning("Failed to read frame from camera")
                    time.sleep(0.01)  # Small delay before retry
        finally:
            cap.release()
            locallogs.debug("Camera released, putting sentinel in queue")
            try:
                frame_queue.put(None, block=True, timeout=QUEUE_TIMEOUT_SECONDS)
            except queue.Full:
                locallogs.error("Frame queue full, could not put sentinel")

    def capture(
        self,
        output_dir: Optional[str] = None,
        show_video: bool = True,
    ) -> None:
        """
        Start frame capture and recording.

        Args:
            output_dir: Output directory (defaults to config.output_dir)
            show_video: If True, display video preview window
        """
        self.terminate.clear()

        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create video writer with Unix timestamp filename
        timestamp = int(time.time())  # seconds (for filename)
        video_path = Path(output_dir) / f"{timestamp}.avi"
        csv_path = Path(output_dir) / f"{timestamp}.csv"

        # Get actual resolution and fps (may differ from requested)
        # We'll get these from the first frame
        actual_fps = self.config.fps
        actual_width = self.config.frame_width
        actual_height = self.config.frame_height

        # Determine pixel format from config or auto-detect based on codec
        pix_fmt = determine_pix_fmt(self.config.codec, self.config.pix_fmt)

        writer = VideoWriter(
            path=video_path,
            fps=actual_fps,
            output_dict={
                "-vcodec": self.config.codec,
                "-f": "avi",
                "-pix_fmt": pix_fmt,
            },
        )

        csv_writer = BufferedCSVWriter(
            file_path=csv_path,
            header=["frame_index", "unix_time"],
            buffer_size=CSV_BUFFER_SIZE,
        )

        shared_resource_manager = multiprocessing.Manager()
        frame_queue = shared_resource_manager.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

        p_camera = multiprocessing.Process(
            target=self._camera_recv,
            args=(frame_queue,),
            name="camera_recv",
        )

        p_camera.start()

        self.logger.info(f"Recording to {video_path}")
        self.logger.info("Press Ctrl+C to stop recording")

        frames_written = 0
        frame_index = 0
        first_frame = True
        start_time = None
        last_fps_log_time = None
        frames_in_window = 0
        writer_used = False

        try:
            for frame_data in exact_iter(frame_queue.get, None):
                if frame_data is None:
                    # Early termination signal from camera process (camera failed)
                    if frames_written == 0:
                        raise RuntimeError(
                            "Camera failed to initialize or read frames. "
                            "Please check camera connection and settings."
                        )
                    break

                frame, unix_time = frame_data

                # Get actual dimensions from first frame and initialize timing
                if first_frame:
                    actual_height, actual_width = frame.shape[:2]
                    self.logger.info(
                        f"Resolution: {actual_width}x{actual_height} @ {actual_fps}fps"
                    )
                    # Start FPS counting from the first grabbed frame
                    start_time = unix_time
                    last_fps_log_time = unix_time
                    first_frame = False

                # Convert frame based on codec
                frame_out = convert_frame_for_codec(frame, self.config.codec)

                # Write frame to video
                writer.write_frame(frame_out)
                writer_used = True

                # Write frame metadata to CSV
                csv_writer.append(
                    {
                        "frame_index": frame_index,
                        "unix_time": unix_time,
                    }
                )

                frames_written += 1
                frame_index += 1
                frames_in_window += 1

                # Log FPS at regular intervals (FPS for the last window)
                current_time = time.time()
                window_elapsed = current_time - last_fps_log_time
                if window_elapsed >= FPS_LOG_INTERVAL_SECONDS:
                    fps = frames_in_window / window_elapsed
                    total_elapsed = current_time - start_time
                    self.logger.info(
                        f"\nFPS:\t{fps:.2f}\nFrames:\t{frames_written} \n"
                        f"Time:\t{total_elapsed:.1f}s \n"
                    )
                    last_fps_log_time = current_time
                    frames_in_window = 0

                # Show preview
                if show_video:
                    try:
                        cv2.imshow("Recording", frame)
                        cv2.waitKey(1)
                    except cv2.error as e:
                        self.logger.exception(f"Error displaying frame: {e}")

        except KeyboardInterrupt:
            self.logger.info("Recording stopped by user (Ctrl+C)")
            self.terminate.set()
        except Exception as e:
            self.logger.exception(f"Error during capture: {e}")
            self.terminate.set()
        finally:
            # Wait for camera process to finish
            self.terminate.set()
            p_camera.join(timeout=5)
            if p_camera.is_alive():
                self.logger.warning("Termination timeout: force terminating camera process")
                p_camera.terminate()
                p_camera.join()

            # Close writers (only if used)
            if writer_used:
                try:
                    writer.close()
                except AttributeError as e:
                    # FFmpegWriter may not have _proc if no frames were written
                    self.logger.warning(f"Error closing video writer: {e}")
            else:
                # Remove empty video file if no frames were written
                if video_path.exists():
                    video_path.unlink()
            csv_writer.close()

            if show_video:
                cv2.destroyAllWindows()
                cv2.waitKey(100)

            self.logger.info(f"Saved recording to {video_path} ({frames_written} frames written)")

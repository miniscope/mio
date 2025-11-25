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
from mio.io import BufferedCSVWriter, VideoWriter
from mio.models.usbcam import USBCameraRecordingConfig
from mio.types import ConfigSource
from mio.utils import exact_iter

# Constants
FPS_LOG_INTERVAL_SECONDS = 5.0
FRAME_QUEUE_MAXSIZE = 30
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
    ) -> None:
        """
        Initialize behavior camera capture.

        Args:
            recording_config: Configuration object, config ID, or path to config file
        """
        self.logger = init_logger("behaviorCam")
        self.config = USBCameraRecordingConfig.from_any(recording_config)
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

        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            frame_queue.put(None)
            raise RuntimeError(f"Failed to open camera at index {self.config.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

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
        if self.config.pix_fmt is not None:
            pix_fmt = self.config.pix_fmt
        elif self.config.codec.lower() == "mjpeg":
            pix_fmt = "yuvj420p"  # YUV color format for MJPEG
        elif self.config.codec.lower() == "rawvideo":
            pix_fmt = "gray"  # Grayscale for rawvideo
        else:
            pix_fmt = "yuv420p"  # Default YUV format for other codecs

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
        start_time = time.time()
        last_fps_log_time = start_time
        frames_in_window = 0

        try:
            for frame_data in exact_iter(frame_queue.get, None):
                if frame_data is None:
                    break

                frame, unix_time = frame_data

                # Get actual dimensions from first frame
                if first_frame:
                    actual_height, actual_width = frame.shape[:2]
                    self.logger.info(
                        f"Resolution: {actual_width}x{actual_height} @ {actual_fps}fps"
                    )
                    first_frame = False

                # Convert frame based on codec
                if self.config.codec.lower() == "rawvideo":
                    # Grayscale for rawvideo
                    if len(frame.shape) == 3:
                        frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame_out = frame
                else:
                    # RGB for color codecs (mjpeg, libx264, etc.)
                    if len(frame.shape) == 3:
                        frame_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        # If grayscale, convert to RGB
                        frame_out = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                # Write frame to video
                writer.write_frame(frame_out)

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
                        f"FPS: {fps:.2f} | Frames: {frames_written} | "
                        f"Time: {total_elapsed:.1f}s"
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

            # Close writers
            writer.close()
            csv_writer.close()

            if show_video:
                cv2.destroyAllWindows()
                cv2.waitKey(100)

            self.logger.info(f"Saved recording to {video_path} ({frames_written} frames written)")

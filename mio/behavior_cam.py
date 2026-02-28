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
import numpy as np

from mio import init_logger
from mio.devices.usbcam import convert_frame_for_codec, open_camera
from mio.exceptions import EndOfRecordingException
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
                capture_format=self.config.format,
            )
        except RuntimeError:
            frame_queue.put(None)
            raise

        locallogs.info("Camera opened, starting frame capture")

        try:
            while not self.terminate.is_set():
                try:
                    ret, frame = cap.read()
                except EndOfRecordingException:
                    locallogs.info("End of recorded data")
                    break
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
        capture_binary: Optional[Path] = None,
    ) -> None:
        """
        Start frame capture and recording.

        Args:
            output_dir: Output directory (defaults to config.output_dir)
            show_video: If True, display video preview window
            capture_binary: If set, save raw frames and timestamps to this ``.npz`` path
        """
        self.terminate.clear()

        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create video writer with Unix timestamp filename
        timestamp = int(time.time())  # seconds (for filename)

        # Determine container format based on codec
        if self.config.codec in ("libx264", "h264"):
            video_ext = ".mp4"
            container_format = "mp4"
        else:
            video_ext = ".avi"
            container_format = "avi"

        video_path = Path(output_dir) / f"{timestamp}{video_ext}"
        csv_path = Path(output_dir) / f"{timestamp}.csv"

        # Get actual resolution and fps (may differ from requested)
        # We'll get these from the first frame
        actual_fps = self.config.fps
        actual_width = self.config.frame_width
        actual_height = self.config.frame_height

        # Build output_dict (pix_fmt only used by skvideo backend, cv2 ignores it)
        output_dict = {
            "-vcodec": self.config.codec,
            "-f": container_format,
            "-vsync": "0",  # Disable frame sync - write frames as-is (same as StreamDaq)
        }
        # Only add pix_fmt for skvideo backend (cv2 doesn't use it)
        if self.config.backend == "skvideo" and self.config.pix_fmt is not None:
            output_dict["-pix_fmt"] = self.config.pix_fmt

        writer = VideoWriter(
            path=video_path,
            fps=actual_fps,
            output_dict=output_dict,
            backend=self.config.backend,
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
        self.logger.info("Press 'q' to stop recording")

        frames_written = 0
        first_frame = True
        start_time = None
        last_fps_log_time = None
        frames_in_window = 0
        writer_used = False
        binary_frames = [] if capture_binary else None
        binary_timestamps = [] if capture_binary else None

        try:
            for frame_data in exact_iter(frame_queue.get, None):
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
                        "frame_index": frames_written,
                        "unix_time": unix_time,
                    }
                )

                if capture_binary:
                    binary_frames.append(frame)
                    binary_timestamps.append(unix_time)

                frames_written += 1
                frames_in_window += 1

                # Log FPS at regular intervals (FPS for the last window)
                current_time = time.time()
                window_elapsed = current_time - last_fps_log_time
                if window_elapsed >= FPS_LOG_INTERVAL_SECONDS:
                    measured_fps = frames_in_window / window_elapsed
                    total_elapsed = current_time - start_time
                    self.logger.info(
                        f"\nFPS:\t{measured_fps:.2f}\nFrames:\t{frames_written} \n"
                        f"Time:\t{total_elapsed:.1f}s \n"
                    )
                    last_fps_log_time = current_time
                    frames_in_window = 0

                # Show preview and check for 'q' key
                if show_video:
                    try:
                        cv2.imshow("Recording", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self.logger.info("Recording stopped by user ('q' key)")
                            break
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

            if capture_binary and binary_frames:
                np.savez(
                    capture_binary,
                    frames=np.array(binary_frames, dtype=np.uint8),
                    timestamps=np.array(binary_timestamps, dtype=np.float64),
                )
                self.logger.info(
                    f"Saved binary data to {capture_binary} ({len(binary_frames)} frames)"
                )

            if show_video:
                cv2.destroyAllWindows()
                cv2.waitKey(100)

            if writer_used:
                self.logger.info(
                    f"Saved recording to {video_path} ({frames_written} frames written)"
                )

                # Verify video frame count matches CSV row count
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if video_frame_count != frames_written:
                        self.logger.warning(
                            f"Frame count mismatch: video has {video_frame_count} frames "
                            f"but CSV has {frames_written} rows"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Could not verify video frame count ({self.config.backend} backend): {e}"
                    )
            else:
                self.logger.warning("No frames were written, recording file removed")

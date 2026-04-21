"""Reading and writing video files"""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from skvideo.io import FFmpegWriter

from mio import init_logger


class VideoWriter:
    """
    Write data to a video file using FFMpegWriter.
    """

    DEFAULT_OUTPUT = {
        "-vcodec": "rawvideo",
        "-f": "avi",
        "-pix_fmt": "gray",
        "-vsync": "0",
    }

    def __init__(
        self,
        path: str | Path,
        fps: int,
        output_dict: dict | None = None,
        force: bool = False,
    ):
        """
        Initialize the VideoWriter object.
        """
        if output_dict is None:
            output_dict = {}
        output_dict = {**self.DEFAULT_OUTPUT, **output_dict}

        input_dict = {"-framerate": str(fps)}

        self.path = Path(path)
        if force:
            self.path.unlink(missing_ok=True)
        elif self.path.exists():
            raise FileExistsError(f"{self.path} exists! use force=True to overwrite.")

        self.writer = FFmpegWriter(
            filename=str(self.path), inputdict=input_dict, outputdict=output_dict
        )

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video file.

        Parameters:
        frame (np.ndarray): The frame to write.
        """
        self.writer.writeFrame(frame)
        return True

    def close(self) -> None:
        """
        Close the video file.
        """
        self.writer.close()


class VideoReader:
    """
    A class to read video files.
    """

    def __init__(self, video_path: str):
        """
        Initialize the VideoReader object.

        Parameters:
        video_path (str): The path to the video file.

        Raises:
        ValueError: If the video file cannot be opened.
        """
        self.video_path = video_path
        self.logger = init_logger("VideoReader")
        self._cap = None

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")

        self.logger.info(f"Opened video at {video_path}")

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def height(self) -> int:
        """
        The height of the video frames.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self) -> int:
        """
        The width of the video frames.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def cap(self) -> cv2.VideoCapture:
        """
        The OpenCV video capture object.
        """
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.video_path))
            self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        return self._cap

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """
        Read frames from the video file along with their index.

        Yields:
        tuple[int, np.ndarray]: The 0-based index and the frame data.
        """
        while self.cap.isOpened():
            # Get frame position BEFORE reading - CAP_PROP_POS_FRAMES returns
            # the 0-based index of the next frame to be captured/decoded
            index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = self.cap.read()
            if not ret:
                break

            self.logger.debug(f"Reading frame {index}")

            yield index, frame

    def read_frame(self, index: int) -> np.ndarray | None:
        """
        Read a frame from the video file.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        """
        Release the video capture object.
        """
        self.cap.release()
        self._cap = None

    def __del__(self):
        with contextlib.suppress(AttributeError):
            self.release()

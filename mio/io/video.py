"""Reading and writing video files"""

import contextlib
from pathlib import Path
from typing import Iterator, Tuple, Union

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
        "-filter:v": "format=gray",
    }

    def __init__(
        self,
        path: Union[str, Path],
        fps: int,
        output_dict: Union[dict, None] = None,
    ):
        """
        Initialize the VideoWriter object.
        """
        if output_dict is None:
            output_dict = {}
        output_dict = {**self.DEFAULT_OUTPUT, **output_dict}
        output_dict["-r"] = str(fps)

        self.writer = FFmpegWriter(filename=str(path), outputdict=output_dict)

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video file.

        Parameters:
        frame (np.ndarray): The frame to write.
        """
        self.writer.writeFrame(frame)

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
        return self._cap

    def read_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Read frames from the video file along with their index.

        Yields:
        Tuple[int, np.ndarray]: The index and the next frame in the video.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.logger.debug(f"Reading frame {index}")

            yield index, frame

    def release(self) -> None:
        """
        Release the video capture object.
        """
        self.cap.release()
        self._cap = None

    def __del__(self):
        with contextlib.suppress(AttributeError):
            self.release()

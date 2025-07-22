"""
I/O functions for files.
"""

import atexit
import contextlib
import csv
from pathlib import Path
from typing import Any, Iterator, List, Tuple, Union

import cv2
import numpy as np
from skvideo.io import FFmpegWriter

from mio.logging import init_logger


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


class BufferedCSVWriter:
    """
    Write data to a CSV file in buffered mode.

    Parameters
    ----------
    file_path : Union[str, Path]
        The file path for the CSV file.
    buffer_size : int, optional
        The number of rows to buffer before writing to the file (default is 100).

    Attributes
    ----------
    file_path : Path
        The file path for the CSV file.
    buffer_size : int
        The number of rows to buffer before writing to the file.
    buffer : list
        The buffer for storing rows before writing.
    """

    def __init__(self, file_path: Union[str, Path], buffer_size: int = 100):
        self.file_path: Path = Path(file_path)
        self.buffer_size = buffer_size
        self.buffer = []
        self.logger = init_logger("BufferedCSVWriter")

        # Ensure the buffer is flushed when the program exits
        atexit.register(self.flush_buffer)

    def append(self, data: List[Any]) -> None:
        """
        Append data (as a list) to the buffer.

        Parameters
        ----------
        data : List[Any]
            The data to be appended.
        """
        data = [int(value) if isinstance(value, np.generic) else value for value in data]
        self.buffer.append(data)
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self) -> None:
        """
        Write all buffered rows to the CSV file.
        """
        if not self.buffer:
            return

        try:
            with open(self.file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.buffer)
                self.buffer.clear()
        except Exception as e:
            # Handle exceptions, e.g., log them
            self.logger.error(f"Failed to write to file {self.file_path}: {e}")

    def close(self) -> None:
        """
        Close the CSV file and flush any remaining data.
        """
        self.flush_buffer()
        # Prevent flush_buffer from being called again at exit
        atexit.unregister(self.flush_buffer)

    def __del__(self):
        self.close()

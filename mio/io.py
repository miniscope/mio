"""
I/O functions for SD card and external files.
"""

import atexit
import contextlib
import csv
from pathlib import Path
from typing import BinaryIO, Iterator, Literal, Optional, Tuple, Union, overload

import cv2
import numpy as np
from skvideo.io import FFmpegWriter
from tqdm import tqdm

from mio.exceptions import EndOfRecordingException, ReadHeaderException
from mio.logging import init_logger
from mio.models.data import Frame
from mio.models.sdcard import SDBufferHeader, SDConfig, SDLayout
from mio.types import ConfigSource


def ensure_gray_uint8(frame: np.ndarray, where: str) -> np.ndarray:
    """Normalize a frame to 2D uint8 grayscale and validate.

    Args:
        frame: Input frame array
        where: Context string for error messages

    Returns:
        np.ndarray: 2D uint8 grayscale frame
    """
    if frame is None:
        raise ValueError(f"{where} received None frame")
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if isinstance(frame, np.ndarray) and frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    if not isinstance(frame, np.ndarray) or frame.ndim != 2 or frame.dtype != np.uint8:
        raise ValueError(
            f"{where} expects 2D uint8 grayscale; got shape={getattr(frame, 'shape', None)} "
            f"dtype={getattr(frame, 'dtype', None)}"
        )
    return frame


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
        path: Union[str, Path],
        fps: int,
        output_dict: Union[dict, None] = None,
    ):
        """
        Initialize the VideoWriter object.
        """
        self.logger = init_logger("VideoWriter")
        if output_dict is None:
            output_dict = {}
        output_dict = {**self.DEFAULT_OUTPUT, **output_dict}

        input_dict = {"-framerate": str(fps)}

        self.writer = FFmpegWriter(filename=str(path), inputdict=input_dict, outputdict=output_dict)

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the video file.

        Parameters:
        frame (np.ndarray): The frame to write.

        Returns:
        bool: True if the frame write was attempted and did not raise anything.
        """
        # Enforce grayscale uint8 (with normalization)
        frame = ensure_gray_uint8(frame, "VideoWriter")

        self.writer.writeFrame(frame)
        return True

    def close(self) -> None:
        """
        Close the video file.
        """
        try:
            # FFmpegWriter may not start (_proc is None) if no frames were written
            if hasattr(self.writer, "_proc") and getattr(self.writer, "_proc", None) is not None:
                self.writer.close()
            else:
                self.logger.warning(
                    "VideoWriter closed without any frames written; nothing to finalize."
                )
        except AttributeError:
            self.logger.warning("VideoWriter backend not initialized; nothing to close.")


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
            # Prefer native output when available (avoid implicit BGR conversion)
            with contextlib.suppress(Exception):
                self._cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
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

            frame = ensure_gray_uint8(frame, "VideoReader")

            index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.logger.debug(f"Reading frame {index}")

            yield index, frame

    def read_frame(self, index: int) -> np.ndarray:
        """
        Read a frame from the video file.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return frame
        frame = ensure_gray_uint8(frame, "VideoReader")
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


class BufferedCSVWriter:
    """
    Write data to a CSV file in buffered mode.

    Parameters
    ----------
    file_path : Union[str, Path]
        The file path for the CSV file.
    headers : list[str]
        Headers for csv - determine the order of columns and allowable values
        passed to :meth:`.append`
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

    def __init__(self, file_path: Union[str, Path], header: list[str], buffer_size: int = 100):
        self.file_path: Path = Path(file_path)
        self.header = header
        self.buffer_size = buffer_size
        self.buffer = []
        self.logger = init_logger("BufferedCSVWriter")

        # write header in first row
        self.buffer.append(self.header)

        # Ensure the buffer is flushed when the program exits
        atexit.register(self.flush_buffer)

    def append(self, data: dict) -> None:
        """
        Append data (as a list) to the buffer.

        Parameters
        ----------
        data : dict
            The data to be appended.
            Rows are constructed and columns are ordered according to `header` -
            keys that are not in `header` are ignored, and missing keys are `None`
        """
        row = [data.get(key) for key in self.header]
        self.buffer.append(row)
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


class SDCard:
    """
    I/O for data on an SDCard

    an instance of :class:`.sdcard.SDLayout` (typically in :mod:`.formats` )
    configures how the data is laid out on the SD card. This class makes the i/o
    operations abstract over multiple layouts

    Args:
        drive (str, :class:`pathlib.Path`): Path to the SD card drive
        layout (:class:`.sdcard.SDLayout`): A layout configuration for an SD card

    """

    def __init__(
        self, drive: Union[str, Path], layout: Union[SDLayout, ConfigSource] = "wirefree-sd-layout"
    ):
        self.drive = drive
        self.layout = SDLayout.from_any(layout)
        self.logger = init_logger("SDCard")

        # Private attributes used when the file reading context is entered
        self._config = None  # type: Optional[SDConfig]
        self._f = None  # type: Optional[BinaryIO]
        self._frame = None  # type: Optional[int]
        self._frame_count = None  # type: Optional[int]
        self._array = None  # type: Optional[np.ndarray]
        """
        n_pix x 1 array used to store pixels while reading buffers
        """
        self.positions = {}
        """
        A mapping between frame number and byte position in the video that makes for 
        faster seeking :)
        
        As we read, we store the locations of each frame before reading it. Later, 
        we can assign to `frame` to seek back to those positions. Assigning to `frame` 
        works without caching position, but has to manually iterate through each frame.
        """

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def config(self) -> SDConfig:
        """
        Read configuration from SD Card
        """
        if self._config is None:
            with open(self.drive, "rb") as sd:
                sd.seek(self.layout.sectors.config_pos, 0)
                configSectorData = np.frombuffer(sd.read(self.layout.sectors.size), dtype=np.uint32)

            self._config = SDConfig(
                **{
                    k: configSectorData[v]
                    for k, v in self.layout.config.model_dump().items()
                    if v is not None
                }
            )

        return self._config

    @property
    def position(self) -> Optional[int]:
        """
        When entered as context manager, the current position of the internal file
        descriptor
        """
        if self._f is None:
            return None

        return self._f.tell()

    @property
    def frame(self) -> Optional[int]:
        """
        When reading, the number of the frame that would be read if we were to call
        :meth:`.read`
        """
        if self._f is None:
            return None

        return self._frame

    @frame.setter
    def frame(self, frame: int) -> None:
        """
        Seek to a specific frame

        Arguments:
            frame (int): The frame to seek to!
        """
        if self._f is None:
            raise RuntimeError(
                "Havent entered context manager yet! Cant change position without that!"
            )

        if frame == self.frame:
            return

        if frame in self.positions:
            self._f.seek(self.positions[frame], 0)
            self._frame = frame
            return
        else:
            # TODO find the nearest position we do have
            pass

        if frame < self.frame:
            # hard to go back, esp if we haven't already been here
            # (we should have stashed the position)
            # just go to start of data and seek like normally (next case)
            self._f.seek(self.layout.sectors.data_pos, 0)
            self._frame = 0

        if frame > self.frame:
            for _ in range(frame - self.frame):
                self.skip()

    @property
    def frame_count(self) -> int:
        """
        Total number of frames in recording.

        Inferred from :class:`~.sdcard.SDConfig.n_buffers_recorded` and
        reading a single frame to get the number of buffers per frame.
        """
        if self._frame_count is None:
            if self._f is None:
                with self as self_open:
                    frame = self_open.read(return_header=True)
                    headers = frame.headers

            else:
                # If we're already open, great, just return to the last frame
                last_frame = self.frame
                # Go one frame back in case we are at the end of the data
                self.frame = max(last_frame - 1, 0)
                frame = self.read(return_header=True)
                headers = frame.headers
                self.frame = last_frame

            self._frame_count = int(
                np.ceil(
                    (self.config.n_buffers_recorded + self.config.n_buffers_dropped) / len(headers)
                )
            )

        # if we have since read more frames than should be there, we update the
        # frame count with a warning
        max_pos = np.max(list(self.positions.keys()))
        if max_pos > self._frame_count:
            self.logger.warning(
                "Got more frames than indicated in card header, expected "
                f"{self._frame_count} but got {max_pos}"
            )
            self._frame_count = int(max_pos)

        return self._frame_count

    # --------------------------------------------------
    # Context Manager methods
    # --------------------------------------------------

    def __enter__(self) -> "SDCard":
        if self._f is not None:
            raise RuntimeError("Cant enter context, and open the file twice!")

        # init private attrs
        # create an empty frame to hold our data!
        self._array = np.zeros((self.config.width * self.config.height, 1), dtype=np.uint8)
        self._pixel_count = 0
        self._last_buffer_n = 0
        self._frame = 0

        self._f = open(self.drive, "rb")  # noqa: SIM115 - this is a context handler
        # seek to the start of the data
        self._f.seek(self.layout.sectors.data_pos, 0)
        # store the 0th frame position
        self.positions[0] = self.layout.sectors.data_pos

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self._f.close()
        self._f = None
        self._frame = 0

    # --------------------------------------------------
    # read methods
    # --------------------------------------------------
    def _read_data_header(self, sd: BinaryIO) -> SDBufferHeader:
        """
        Given an already open file buffer opened in bytes mode,
         seeked to the start of a frame, read the data header
        """

        # read one word first, I think to get the size of the rest of the header,
        # that sort of breaks the abstraction
        # (it assumes the buffer length is always at position 0)
        # but we'll roll with it for now
        dataHeader = np.frombuffer(sd.read(self.layout.word_size), dtype=np.uint32)
        dataHeader = np.append(
            dataHeader,
            np.frombuffer(
                sd.read((dataHeader[self.layout.buffer.length] - 1) * self.layout.word_size),
                dtype=np.uint32,
            ),
        )

        # use construct because we're already sure these are ints from the numpy casting
        # https://docs.pydantic.dev/latest/usage/models/#creating-models-without-validation
        try:
            header = SDBufferHeader.from_format(dataHeader, self.layout.buffer, construct=True)
        except IndexError as e:
            raise ReadHeaderException(
                "Could not read header, expected header to have "
                f"{len(self.layout.buffer.model_dump().keys())} fields, "
                f"got {len(dataHeader)}. Likely mismatch between specified "
                "and actual SD Card layout or reached end of data.\n"
                f"Header Data: {dataHeader}"
            ) from e

        return header

    def _n_frame_blocks(self, header: SDBufferHeader) -> int:
        """
        Compute the number of blocks for a given frame buffer

        Not sure how this works!
        """
        n_blocks = int(
            (
                header.data_length
                + (header.length * self.layout.word_size)
                + (self.layout.sectors.size - 1)
            )
            / self.layout.sectors.size
        )
        return n_blocks

    def _read_size(self, header: SDBufferHeader) -> int:
        """
        Compute the number of bytes to read for a given buffer

        Not sure how this works with :meth:`._n_frame_blocks`, but keeping
        them separate in case they are separable actions for now
        """
        n_blocks = self._n_frame_blocks(header)
        read_size = (n_blocks * self.layout.sectors.size) - (header.length * self.layout.word_size)
        return read_size

    def _read_buffer(self, sd: BinaryIO, header: SDBufferHeader) -> np.ndarray:
        """
        Read a single buffer from a frame.

        Each frame has several buffers, so for a given frame we read them until we
        get another that's zero!
        """
        data = np.frombuffer(sd.read(self._read_size(header)), dtype=np.uint8)
        return data

    def _trim(self, data: np.ndarray, expected_size: int) -> np.ndarray:
        """
        Trim or pad an array to match an expected size
        """
        if data.shape[0] != expected_size:
            self.logger.warning(
                f"Frame: {self._frame}: Expected buffer data length: {expected_size}, "
                f"got data with shape {data.shape}. "
                "Padding to expected length",
                stacklevel=1,
            )

            # trim if too long
            if data.shape[0] > expected_size:
                data = data[0:expected_size]
            # pad if too short
            else:
                data = np.pad(data, (0, expected_size - data.shape[0]))

        return data

    @overload
    def read(self, return_header: Literal[True] = True) -> Frame: ...

    @overload
    def read(self, return_header: Literal[False] = False) -> np.ndarray: ...

    def read(self, return_header: bool = False) -> Union[np.ndarray, Frame]:
        """
        Read a single frame

        Arguments:
            return_header (bool): If `True`, return headers from individual buffers
                (default `False`)

        Return:
            :class:`numpy.ndarray` ,
            or a tuple(ndarray, List[:class:`~.SDBufferHeader`]) if `return_header`
            is `True`
        """
        if self._f is None:
            raise RuntimeError(
                "File is not open! Try entering the reader context by using it like "
                "`with sdcard:`"
            )

        self._array[:] = 0
        pixel_count = 0
        last_buffer_n = 0
        headers = []
        while True:
            # stash position before reading header
            last_position = self._f.tell()
            try:
                header = self._read_data_header(self._f)
            except ValueError as e:
                if "read length must be non-negative" in str(e):
                    # end of file! Value error thrown because the dataHeader will be
                    # blank,  and thus have a value of 0 for the header size, and we
                    # can't read 0 from the card.
                    self._f.seek(last_position, 0)
                    raise EndOfRecordingException("Reached the end of the video!") from None
                else:
                    raise e
            except IndexError as e:
                if "index 0 is out of bounds for axis 0 with size 0" in str(e):
                    # end of file if we are reading from a disk image without any
                    # additional space on disk
                    raise EndOfRecordingException("Reached the end of the video!") from None
                else:
                    raise e
            except ReadHeaderException as e:
                # if we are on the last frame, normal! signal end of iteration
                if self._frame == self.frame_count - 1:
                    raise EndOfRecordingException("Reached the end of the video!") from None
                else:
                    raise e

            if header.frame_buffer_count == 0 and last_buffer_n > 0:
                # we are in the next frame!
                # rewind to the beginning of the header, and return
                # the last_position is the start of the header for this frame
                self._f.seek(last_position, 0)
                self._frame += 1
                self.positions[self._frame] = last_position
                frame = np.reshape(self._array, (self.config.width, self.config.height))
                if return_header:
                    return Frame.model_construct(frame=frame, headers=headers)
                else:
                    return frame

            # grab buffer data and stash
            headers.append(header)
            data = self._read_buffer(self._f, header)
            data = self._trim(data, header.data_length)
            self._array[pixel_count : pixel_count + header.data_length, 0] = data
            pixel_count += header.data_length
            last_buffer_n = header.frame_buffer_count

    # --------------------------------------------------
    # Write methods
    # --------------------------------------------------

    def to_video(
        self,
        path: Union[Path, str],
        fourcc: Literal["GREY", "mp4v", "XVID"] = "GREY",
        isColor: bool = False,
        force: bool = False,
        progress: bool = True,
    ) -> None:
        """
        Save contents of SD card to video with opencv

        Args:
            path (:class:`pathlib.Path`): Output video path, with video extension
                ``.avi`` or ``.mp4``
            fourcc (str): FourCC code used with opencv. Other codecs may be available
                depending on your opencv installation, but by default opencv supports
                one of:

                * ``GREY`` (default)
                * ``mp4v``
                * ``XVID``

            isColor (bool): Indicates whether output video is in color
                (default: `False`)
            force (bool): If `True`, overwrite output video if one already exists
                (default: `False`)
            progress (bool): If `True` (default) show progress bar.
        """
        path = Path(path)
        if path.exists() and not force:
            raise FileExistsError(
                f"{str(path)} already exists, not overwriting. " "Use force=True to overwrite."
            )

        if path.suffix == ".mp4" and fourcc.lower() == "grey":
            self.logger.warning("Cannot use .mp4 with GREY fourcc code. Using .avi instead")
            path = path.with_suffix(".avi")
        elif path.suffix == ".avi" and fourcc.lower() == "mp4v":
            self.logger.warning("Cannot use .avi with mp4v fourcc code, using .mp4 instead")
            path = path.with_suffix(".mp4")

        if progress:
            pbar = tqdm(total=self.frame_count)

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.config.fs,
            (self.config.width, self.config.height),
            isColor=isColor,
        )

        # wrap in try block so we always close video writer class
        try:
            # Open video context manager
            with self:
                # catch the stop iteration signal
                try:
                    while True:
                        # this is sort of an awkward stack, should probably make a
                        # generator version of `read`
                        frame = self.read(return_header=False)
                        writer.write(frame)

                        if progress:
                            pbar.update()

                except StopIteration:
                    # end of the video!
                    pass

        finally:
            writer.release()
            if progress:
                pbar.close()

    def to_img(
        self,
        path: Optional[Union[Path, str]],
        frame: Optional[int] = None,
        force: bool = False,
        chunk_size: int = 1e6,
        progress: bool = True,
    ) -> None:
        """
        Create a new disk image that is truncated to the actual size of the video data

        Typically, making disk images using dd or other tools will create an image
        file that is the full size of the media it's stored on. Rather than
        sending a bunch of 30GB image files around, we can instead create an
        image that is truncated to just the size of the data that has actually been recorded

        Args:
            path (:class:`pathlib.Path`): Path to write .img file to
            frame (int): Optional, if present only write the first n frames. If
                ``None`` , write all frames
            force (bool): If ``True``, overwrite an existing file. If ``False`` , (default)
                don't.
            chunk_size (int): Number of bytes to read/write at once (default ``1e6`` )
            progress (bool): If ``True`` (default), show progress bar
        """
        path = Path(path)
        if path.exists() and not force:
            raise FileExistsError("File exists, use force=True to overwrite")

        path.parent.mkdir(parents=True, exist_ok=True)

        start_frame = self.frame

        if frame is None:
            frame = self.frame_count - 1

        # read it to move the file position to the end of the frame
        with self:
            # go to one before the requested frame to make sure we have its position
            self.frame = frame - 1
            # read advances us to the end of the requested frame -- where we want to truncate
            with contextlib.suppress(EndOfRecordingException):
                _ = self.read(return_header=True)
            # now we can take the diff of the positions to get the expected offset
            # for the end of the last frame
            # these should be populated from the read and frame reposition
            diff = self.positions[frame] - self.positions[frame - 1]
            final_position = self._f.tell() + diff

        # try block to ensure closing pbar
        pos = 0
        pbar = tqdm(total=final_position) if progress else None

        try:
            with open(self.drive, "rb") as readfile, open(path, "wb") as writefile:
                while pos < final_position:
                    readsize = int(np.min([final_position - pos, chunk_size]))
                    data = readfile.read(readsize)
                    writefile.write(data)
                    pos = readfile.tell()
                    if progress:
                        pbar.update(readsize)

        finally:
            if progress:
                pbar.close()

        # return frame to the place it was before we wrote
        if start_frame is not None:
            with self:
                self.frame = start_frame

    # --------------------------------------------------
    # General Methods
    # --------------------------------------------------

    def skip(self) -> None:
        """
        Skip a frame

        Read the buffer headers to determine buffer sizes and just seek ahead
        """
        if self._f is None:
            raise RuntimeError(
                "File is not open! Try entering the reader context by using it like "
                "`with sdcard:`"
            )

        last_position = self._f.tell()
        header = self._read_data_header(self._f)

        if header.frame_buffer_count != 0:
            self._f.seek(last_position, 0)
            raise RuntimeError(
                "Did not start at the first buffer of a frame! Something is wrong with "
                "the way seeking is working. Rewound to where we started"
            )

        while True:
            # jump ahead according to the last header we read
            read_size = self._read_size(header)
            self._f.seek(read_size, 1)

            # stash position before reading next buffer header
            last_position = self._f.tell()
            header = self._read_data_header(self._f)

            # if the frame is over, return to the start of the buffer and break,
            # incrementing the current frame count
            if header.frame_buffer_count == 0:
                self._f.seek(last_position, 0)
                self._frame += 1
                self.positions[self.frame] = last_position
                break

    def check_valid(self) -> bool:
        """
        Checks that the header sector has the appropriate write keys in it

        Returns:
            bool - True if valid, False if not
        """
        with open(self.drive, "rb") as sd:
            sd.seek(self.layout.sectors.header_pos, 0)
            headerSectorData = np.frombuffer(sd.read(self.layout.sectors.size), dtype=np.uint32)

            valid = False
            if (
                headerSectorData[0] == self.layout.write_key0
                and headerSectorData[1] == self.layout.write_key1
                and headerSectorData[2] == self.layout.write_key2
                and headerSectorData[3] == self.layout.write_key3
            ):
                valid = True

        return valid

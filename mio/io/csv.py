"""
I/O functions for files.
"""

import atexit
import csv
from pathlib import Path
from typing import Any

from noob import deinit_method
from noob.utils import resolve_python_identifier
from pydantic import BaseModel

from mio.devices.base import BufferHeader
from mio.logging import init_logger
from mio.types import PythonIdentifier


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
    force: bool
        If True, remove any existing file and start a new one

    Attributes
    ----------
    file_path : Path
        The file path for the CSV file.
    buffer_size : int
        The number of rows to buffer before writing to the file.
    buffer : list
        The buffer for storing rows before writing.
    """

    def __init__(
        self,
        file_path: str | Path,
        header: list[str] | PythonIdentifier,
        buffer_size: int = 100,
        force: bool = False,
    ):
        if isinstance(header, str):
            header_cls = resolve_python_identifier(header)
            if not issubclass(header_cls, BufferHeader):
                raise TypeError("Header class must be a buffer header")
            header = header_cls.csv_header_cols()
        self.file_path: Path = Path(file_path)
        self.header = header
        self.buffer_size = buffer_size
        self.buffer = []
        self.logger = init_logger("BufferedCSVWriter")

        if force:
            self.file_path.unlink(missing_ok=True)

        # write header in first row
        self.buffer.append(self.header)

        # Ensure the buffer is flushed when the program exits
        atexit.register(self.flush_buffer)

    def process(self, data: dict | BaseModel, **kwargs: Any) -> None:
        """
        Append data (as a list) to the buffer.

        Parameters
        ----------
        data : dict
            The data to be appended.
            Rows are constructed and columns are ordered according to `header` -
            keys that are not in `header` are ignored, and missing keys are `None`
        """
        if isinstance(data, BaseModel):
            data = data.model_dump()
        data = {**data, **kwargs}
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

    @deinit_method
    def close(self) -> None:
        """
        Close the CSV file and flush any remaining data.
        """
        self.flush_buffer()
        # Prevent flush_buffer from being called again at exit
        atexit.unregister(self.flush_buffer)

    def __del__(self):
        self.close()

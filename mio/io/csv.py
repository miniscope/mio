"""
I/O functions for files.
"""

import atexit
import csv
from pathlib import Path
from typing import Union

from mio.logging import init_logger


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

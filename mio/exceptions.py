"""
Custom exceptions!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class InvalidSDException(Exception):
    """
    Raised when :class:`.io.SDCard` is used with a drive that doesn't have the
    appropriate WRITE KEYS in its header
    """


class EndOfRecordingException(StopIteration):
    """
    Raised when :class:`.io.SDCard` is at the end of the available recording!
    """


class SDException(Exception):
    """
    Base class for SDcard-specific errors
    """


class ReadHeaderException(SDException, RuntimeError):
    """
    Raised when a given frame's header cannot be read!
    """


class StreamError(RuntimeError):
    """
    Base class for errors while streaming data
    """


class StreamReadError(StreamError):
    """
    Error while reading streaming data from a device
    """


class DeviceError(RuntimeError):
    """
    Base class for errors when communicating with or configuring devices
    """


class DeviceOpenError(DeviceError):
    """
    Error opening a connection to a device
    """


class DeviceConfigurationError(DeviceError):
    """
    Error while configuring a device
    """


class VideoMetadataError(ValueError):
    """
    Raised when video/CSV metadata validation fails.

    Attributes:
        csv_df: The partially-read DataFrame, if available.
    """

    def __init__(self, message: str, csv_df: pd.DataFrame | None = None):
        super().__init__(message)
        self.csv_df = csv_df

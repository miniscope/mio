"""
I/O SDK for UCLA Miniscopes
"""

from importlib import metadata

from mio.logging import init_logger
from mio.models.config import Config

__all__ = [
    "Config",
    "init_logger",
]

try:
    __version__ = metadata.version("mio")
except metadata.PackageNotFoundError:  # pragma: nocover
    __version__ = None

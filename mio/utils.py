"""
The junk drawer my dogs
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Generator, TypeVar, Union

import cv2

if TYPE_CHECKING:
    pass

T = TypeVar("T")


def exact_iter(f: Callable[[], T], sentinel: T) -> Generator[T, None, None]:
    """
    A version of :func:`iter` that compares with `is` rather than `==`
    because truth value of numpy arrays is ambiguous.

    Args:
        f: Function to call repeatedly
        sentinel: Sentinel value to stop iteration when `f()` returns this (compared with `is`)

    Yields:
        Values from `f()` until sentinel is encountered
    """
    while True:
        val = f()
        if val is sentinel:
            break
        else:
            yield val


def hash_file(path: Union[Path, str]) -> str:
    """
    Return the sha256 hash of a file

    Args:
        path (:class:`pathlib.Path`): File to hash
        hash (str): Hash algorithm to use

    Returns:
        str: Hash of file

    References:
        https://stackoverflow.com/a/44873382
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_video(
    path: Union[Path, str],
    method: str = "blake2s",
) -> str:
    """
    Create a hash of a video by digesting the byte string each of its decoded frames.

    Intended to remove variability in video encodings across platforms.

    Args:
        path (:class:`pathlib.Path`): Video file
        method (str): hashing algorithm to use (one of
            :data:`hashlib.algorithms_available` )

    Returns:
        str
    """
    h = hashlib.new(method)

    vid = cv2.VideoCapture(str(path))
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        h.update(frame)  # type: ignore

    return h.hexdigest()

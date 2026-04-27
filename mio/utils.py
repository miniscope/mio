"""
The junk drawer my dogs
"""

import hashlib
from pathlib import Path

import cv2


def hash_file(path: Path | str) -> str:
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
    path: Path | str,
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
    if not Path(path).exists():
        raise FileNotFoundError("No such video exists!")
    h = hashlib.new(method)

    vid = cv2.VideoCapture(str(path))
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        h.update(frame)  # type: ignore

    return h.hexdigest()


def _format_ranges(indices: list[int] | set[int]) -> list[str]:
    """Convert a sorted list of missing frame indices into readable ranges."""
    if not indices:
        return []
    indices = sorted(indices)

    ranges = []
    start = indices[0]
    end = indices[0]

    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = idx
            end = idx

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ranges

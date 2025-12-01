"""
The junk drawer my dogs
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import cv2
import pandas as pd

if TYPE_CHECKING:
    pass


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


def validate_video_metadata_match(
    video_path: Union[Path, str],
) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """
    Validate that a CSV metadata file matches its corresponding video file.

    The CSV file is expected to have the same name as the video file with a .csv extension.

    This function checks:
    1. If CSV file exists (video_path.with_suffix(".csv"))
    2. If CSV has 'reconstructed_frame_index' column
    3. If all frame indices from the video (0 to frame_count-1) exist in the CSV

    Parameters:
    video_path (Union[Path, str]): Path to the video file.

    Returns:
    Tuple[bool, Optional[str], Optional[pd.DataFrame]]: A tuple containing:
        - bool: True if validation passes, False otherwise
        - Optional[str]: Error message if validation fails, None otherwise
        - Optional[pd.DataFrame]: The CSV DataFrame if successfully read, None otherwise
    """
    video_path_obj = Path(video_path)
    csv_path_obj = video_path_obj.with_suffix(".csv")

    # Check if CSV exists
    if not csv_path_obj.exists():
        return False, f"CSV file not found at {csv_path_obj}", None

    # Read CSV
    try:
        df = pd.read_csv(csv_path_obj)
    except Exception as e:
        return False, f"Failed to read CSV file {csv_path_obj}: {e}", None

    # Check if reconstructed_frame_index column exists
    if "reconstructed_frame_index" not in df.columns:
        return (
            False,
            f"CSV file {csv_path_obj} does not have 'reconstructed_frame_index' column",
            None,
        )

    # Get video frame count
    try:
        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            return False, f"Could not open video file {video_path_obj}", None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as e:
        return False, f"Failed to read video file {video_path_obj}: {e}", None

    # Check if all frame indices from video exist in CSV
    expected_frame_indices = set(range(frame_count))
    unique_frame_indices = set(df["reconstructed_frame_index"].unique())
    missing_indices = expected_frame_indices - unique_frame_indices

    if missing_indices:
        # Only report first few missing indices to avoid huge error messages
        missing_list = sorted(missing_indices)
        if len(missing_list) > 10:
            missing_str = (
                f"{', '.join(map(str, missing_list[:10]))}, ... ({len(missing_list)} total missing)"
            )
        else:
            missing_str = ", ".join(map(str, missing_list))
        return (
            False,
            f"Frame indices mismatch: frames {missing_str} not found in CSV. "
            f"Video has {frame_count} frames",
            None,
        )

    return True, None, df

"""
The junk drawer my dogs
"""

import contextlib
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Union

import cv2
import pandas as pd

DEFAULT_PROCESS_DIR = "mio_process"


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
) -> pd.DataFrame:
    """
    Validate that a CSV metadata file matches its corresponding video file.

    Raises :class:`~mio.exceptions.VideoMetadataError` on any validation
    failure.  The exception carries a ``csv_df`` attribute with the partially-
    read DataFrame when the CSV was readable but mismatched.

    Returns the validated DataFrame on success.
    """
    from mio.exceptions import VideoMetadataError

    video_path_obj = Path(video_path)
    csv_path_obj = video_path_obj.with_suffix(".csv")

    if not csv_path_obj.exists():
        raise VideoMetadataError(f"CSV file not found at {csv_path_obj}")

    try:
        df = pd.read_csv(csv_path_obj)
    except Exception as e:
        raise VideoMetadataError(f"Failed to read CSV file {csv_path_obj}: {e}") from e

    if "reconstructed_frame_index" not in df.columns:
        raise VideoMetadataError(
            f"CSV file {csv_path_obj} does not have 'reconstructed_frame_index' column",
            csv_df=df,
        )

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise VideoMetadataError(f"Could not open video file {video_path_obj}", csv_df=df)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    expected_frame_indices = set(range(frame_count))
    unique_frame_indices = set(df["reconstructed_frame_index"].unique())
    missing_indices = expected_frame_indices - unique_frame_indices

    if missing_indices:
        missing_list = sorted(missing_indices)
        if len(missing_list) > 10:
            missing_str = (
                f"{', '.join(map(str, missing_list[:10]))}, ... ({len(missing_list)} total missing)"
            )
        else:
            missing_str = ", ".join(map(str, missing_list))
        raise VideoMetadataError(
            f"Frame indices mismatch: frames {missing_str} not found in CSV. "
            f"Video has {frame_count} frames",
            csv_df=df,
        )

    return df


def validate_frame_count_alignment(
    video_path: Union[Path, str],
) -> None:
    """
    Validate that video frame count matches CSV metadata frame count.

    Raises :class:`~mio.exceptions.VideoMetadataError` on failure.
    """
    from mio.exceptions import VideoMetadataError

    video_path_obj = Path(video_path)
    csv_path_obj = video_path_obj.with_suffix(".csv")

    if not csv_path_obj.exists():
        raise VideoMetadataError(f"CSV file not found at {csv_path_obj}")

    try:
        df = pd.read_csv(csv_path_obj)
    except Exception as e:
        raise VideoMetadataError(f"Failed to read CSV file {csv_path_obj}: {e}") from e

    if "reconstructed_frame_index" not in df.columns:
        raise VideoMetadataError(
            f"CSV file {csv_path_obj} does not have 'reconstructed_frame_index' column"
        )

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise VideoMetadataError(f"Could not open video file {video_path_obj}")
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_csv_index = df["reconstructed_frame_index"].max()
    min_csv_index = df["reconstructed_frame_index"].min()
    unique_indices = set(df["reconstructed_frame_index"].unique())

    expected_max_index = video_frame_count - 1
    expected_indices = set(range(video_frame_count))

    if max_csv_index != expected_max_index:
        raise VideoMetadataError(
            f"Frame count mismatch: video has {video_frame_count} frames "
            f"(indices 0-{expected_max_index}), but CSV max index is {max_csv_index}"
        )

    if min_csv_index != 0:
        raise VideoMetadataError(f"CSV min index is {min_csv_index}, expected 0")

    missing_indices = expected_indices - unique_indices
    if missing_indices:
        missing_list = sorted(missing_indices)
        if len(missing_list) > 10:
            missing_str = (
                f"{', '.join(map(str, missing_list[:10]))}, ... "
                f"({len(missing_list)} total missing)"
            )
        else:
            missing_str = ", ".join(map(str, missing_list))
        raise VideoMetadataError(f"Missing frame indices in CSV: {missing_str}")


def format_missing_frame_ranges(missing_indices: List[int]) -> List[str]:
    """Convert a sorted list of missing frame indices into readable ranges."""
    if not missing_indices:
        return []

    ranges = []
    start = missing_indices[0]
    end = missing_indices[0]

    for idx in missing_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = idx
            end = idx

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ranges


def extract_mismatch_details(
    video_path: Path, is_valid: bool, error_msg: Optional[str], csv_df: Optional[pd.DataFrame]
) -> Optional[dict]:
    """Extract detailed mismatch information from validation result."""
    if is_valid:
        return None

    video_path_obj = Path(video_path)

    if error_msg is None:
        return {"error_type": "unknown"}

    error_lower = error_msg.lower()
    if "not found" in error_lower:
        return {"error_type": "csv_not_found", "error_msg": error_msg}
    elif "failed to read csv" in error_lower:
        return {"error_type": "csv_read_error", "error_msg": error_msg}
    elif "does not have 'reconstructed_frame_index'" in error_lower:
        return {"error_type": "missing_column", "error_msg": error_msg}
    elif "could not open video" in error_lower or "failed to read video" in error_lower:
        return {"error_type": "video_error", "error_msg": error_msg}
    elif "frame indices mismatch" in error_lower or "frames" in error_lower:
        if csv_df is not None and "reconstructed_frame_index" in csv_df.columns:
            video_frame_count = None
            match = re.search(r"Video has (\d+) frames", error_msg)
            if match:
                with contextlib.suppress(ValueError, AttributeError):
                    video_frame_count = int(match.group(1))

            if video_frame_count is None:
                try:
                    cap = cv2.VideoCapture(str(video_path_obj))
                    if cap.isOpened():
                        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                except Exception:
                    pass

            if video_frame_count is not None:
                unique_frame_indices = set(csv_df["reconstructed_frame_index"].unique())
                csv_frame_count = len(unique_frame_indices)
                expected_frame_indices = set(range(video_frame_count))
                missing_indices = sorted(expected_frame_indices - unique_frame_indices)

                return {
                    "error_type": "frame_mismatch",
                    "video_frame_count": video_frame_count,
                    "csv_frame_count": csv_frame_count,
                    "missing_count": len(missing_indices),
                    "missing_indices": missing_indices[:20],
                    "missing_ranges": format_missing_frame_ranges(missing_indices),
                }

        return {"error_type": "frame_mismatch", "error_msg": error_msg}

    return {"error_type": "unknown", "error_msg": error_msg}


def resolve_output_path(
    input_path: Path,
    suffix: str,
    output: str = DEFAULT_PROCESS_DIR,
) -> Path:
    """
    Resolve the output path for process commands.
    Treats output as a directory and generates filename with suffix.
    """
    output_dir = Path(output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"

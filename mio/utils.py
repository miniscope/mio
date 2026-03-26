"""
The junk drawer my dogs
"""

import hashlib
from pathlib import Path

import cv2
import pandas as pd

from mio.exceptions import VideoMetadataError

DEFAULT_PROCESS_DIR = "mio_process"


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
    h = hashlib.new(method)

    vid = cv2.VideoCapture(str(path))
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        h.update(frame)  # type: ignore

    return h.hexdigest()


def validate_video_metadata_match(
    video_path: Path | str,
) -> pd.DataFrame:
    """
    Validate that a CSV metadata file matches its corresponding video file.

    Raises :class:`~mio.exceptions.VideoMetadataError` on any validation
    failure.  The exception carries a ``csv_df`` attribute with the partially-
    read DataFrame when the CSV was readable but mismatched.

    Returns the validated DataFrame on success.
    """
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


def format_missing_frame_ranges(missing_indices: list[int]) -> list[str]:
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


def extract_mismatch_details(video_path: Path, csv_df: pd.DataFrame | None) -> dict:
    """Extract detailed mismatch information for a frame-index validation failure."""
    if csv_df is None or "reconstructed_frame_index" not in csv_df.columns:
        return {"error_type": "frame_mismatch"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error_type": "frame_mismatch"}
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    unique_frame_indices = set(csv_df["reconstructed_frame_index"].unique())
    csv_frame_count = len(unique_frame_indices)
    missing_indices = sorted(set(range(video_frame_count)) - unique_frame_indices)

    return {
        "error_type": "frame_mismatch",
        "video_frame_count": video_frame_count,
        "csv_frame_count": csv_frame_count,
        "missing_count": len(missing_indices),
        "missing_indices": missing_indices[:20],
        "missing_ranges": format_missing_frame_ranges(missing_indices),
    }


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

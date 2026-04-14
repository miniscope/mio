"""
Regression tests for the stitch feature.

Fixtures are trimmed from real recordings.
"""

from pathlib import Path

import cv2
import pandas as pd
import pytest

import numpy as np

from mio.io import VideoWriter
from mio.models.dataset import Recording, Dataset, StitchedRecording
from mio.process.stitch import (
    CandidateFrame,
    StitchRecord,
    stitch,
    _score_edges,
)
from mio.process.video import trim
from mio.utils import hash_video

STITCH_DATA_DIR = Path(__file__).parent.parent / "data" / "stitch"

EXPECTED_STITCHED_VIDEO_HASH = "c8cdf3149f812ae25e6f3f1a876249e4ce118e9a53aa1805e48b995b01f07a91"
EXPECTED_DEBUG_VIDEO_HASH = "856e6e5c538532bd0fcfb942616686a5cd262aadb51dd8796adf5de69215c94b"
EXPECTED_CROP_VIDEO_HASH = "432642b1528fcd9ad553cfb3cc3862bef931301bd11d44dc3c2372fc379fa629"
EXPECTED_CROP_FRAME_COUNT = 30


@pytest.fixture(scope="module")
def recordings() -> dict[str, Recording]:
    return {
        "video1": Recording.from_video(STITCH_DATA_DIR / "video1.avi"),
        "video2": Recording.from_video(STITCH_DATA_DIR / "video2.avi"),
    }


@pytest.fixture(scope="module")
def stitch_result(recordings, tmp_path_factory) -> StitchedRecording:
    """Run stitch once on the trimmed fixtures, return paths to all outputs."""
    output = tmp_path_factory.mktemp("stitch")

    result = stitch(list(recordings.values()), debug_video=True, output_dir=output)

    return result


def test_stitched_video_hash(stitch_result: StitchedRecording):
    """Stitched video content matches expected hash (frame-by-frame blake2s)."""
    assert hash_video(stitch_result.video.path) == EXPECTED_STITCHED_VIDEO_HASH


def test_stitched_video_frame_count(
    stitch_result: StitchedRecording, recordings: dict[str, Recording]
):
    """Stitched video has the expected number of frames (union of frame_nums)."""
    frame_count = stitch_result.video.shape[0]
    expected = np.unique(
        list(recordings["video1"].metadata["frame_num"])
        + list(recordings["video2"].metadata["frame_num"])
    )
    assert frame_count == len(expected)


def test_stitched_csv_contiguous_index(stitch_result: StitchedRecording):
    """reconstructed_frame_index in stitched CSV is contiguous 0..N-1."""
    df = stitch_result.metadata
    indices = sorted(df["reconstructed_frame_index"].unique())
    assert indices == list(range(len(indices)))


def test_stitched_csv_frame_num_range(stitch_result: StitchedRecording, recordings):
    """Stitched CSV covers the full union of frame_nums from both inputs."""
    df = stitch_result.metadata
    frame_nums = np.array(sorted(df["frame_num"].unique()))
    expected = np.unique(
        list(recordings["video1"].metadata["frame_num"])
        + list(recordings["video2"].metadata["frame_num"])
    )
    assert np.array_equal(frame_nums, expected)


def test_debug_video_hash(stitch_result: StitchedRecording):
    """Debug video content matches expected hash."""
    assert hash_video(stitch_result.debug_video.path) == EXPECTED_DEBUG_VIDEO_HASH


def test_score_csv_columns(stitch_result: StitchedRecording):
    """Debug CSV has the columns defined by StitchRecord."""
    assert list(stitch_result.scores.columns) == StitchRecord.header()


def test_score_csv_row_count(stitch_result: StitchedRecording):
    """Debug CSV has one row per frame with pixel differences."""
    assert len(stitch_result.scores) == stitch_result.video.shape[0]


def test_stitch_more_buffers_wins(stitch_result: StitchedRecording):
    """Frame 2530: video1 has 7 buffers, video2 has 8 -> metadata picks video2 (no tie)."""
    df = stitch_result.scores
    row = df[df["frame_num"] == 2530].iloc[0]
    assert pd.isna(row["selected_edge_score"])
    assert row["selected_video"] == "video2"
    assert row["selected_num_buffers"] == 8
    assert row["compare_num_buffers"] == 7


def test_score_csv_edge_scoring_tiebreaker(stitch_result: StitchedRecording):
    """Tied frames use edge scoring: selected frame has higher score (less sharp)."""
    df = stitch_result.scores
    # filter frames where only one video or the other had them
    df = df[~df["compare_video"].isna()]
    # there should be four frames that could be decided on metadata alone
    assert len(df[df["selected_edge_score"].isna()]) == 4
    # for all those that had to use edge scores, the selected should be greater or equal
    edges_scored = df[~df["selected_edge_score"].isna()]
    for _, row in edges_scored.iterrows():
        assert row["selected_edge_score"] >= row["compare_edge_score"]


def test_stitch_without_debug(tmp_path):
    """Stitch produces correct output when debug writers are None."""
    recordings = [
        Recording.from_video(STITCH_DATA_DIR / "video1.avi"),
        Recording.from_video(STITCH_DATA_DIR / "video2.avi"),
    ]

    result = stitch(recordings, output_dir=tmp_path)

    assert hash_video(result.video.path) == EXPECTED_STITCHED_VIDEO_HASH


def test_stitch_padding_tiebreaker(tmp_path, recordings):
    """When buffer counts are equal, less black_padding_px wins."""
    recordings = [
        Recording.from_video(STITCH_DATA_DIR / "video1.avi"),
        Recording.from_video(STITCH_DATA_DIR / "video2.avi"),
    ]
    recordings[1].metadata.loc[
        recordings[1].metadata["frame_num"] == 2520, "black_padding_px"
    ] = 100

    result = stitch(recordings, output_dir=tmp_path)

    # Verify video1 was selected for frame 2520 (less padding)
    df_out = result.scores
    rows = df_out[df_out["frame_num"] == 2520].to_dict("records")
    assert len(rows) == 1
    row = rows[0]
    assert row["selected_video"] == "video1"
    assert row["selected_black_padding"] == 0
    assert row["compare_black_padding"] == 800


def test_trim_video_hash(tmp_path):
    """Cropped video content matches expected hash."""
    out = trim(
        STITCH_DATA_DIR / "video1.avi",
        output_path=tmp_path / "trimped.avi",
        start=10,
        end=10,
    )
    assert hash_video(out) == EXPECTED_CROP_VIDEO_HASH


def test_trim_frame_count(tmp_path):
    """Cropped video has the expected number of frames."""
    out = trim(
        STITCH_DATA_DIR / "video1.avi",
        output_path=tmp_path / "trimped.avi",
        start=10,
        end=10,
    )
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == EXPECTED_CROP_FRAME_COUNT
    cap.release()


def test_trim_csv_valid(tmp_path):
    """Cropped CSV passes video-metadata validation."""
    out = trim(
        STITCH_DATA_DIR / "video1.avi",
        output_path=tmp_path / "trimped.avi",
        start=10,
        end=10,
    )
    recording = Recording.from_video(tmp_path / "trimped.avi")
    assert recording.metadata is not None


def test_trim_csv_renumbered(tmp_path):
    """Cropped CSV has reconstructed_frame_index renumbered to 0-based."""
    out = trim(
        STITCH_DATA_DIR / "video1.avi",
        output_path=tmp_path / "trimped.avi",
        start=10,
        end=10,
    )
    df = pd.read_csv(str(out).replace(".avi", ".csv"))
    indices = sorted(df["reconstructed_frame_index"].unique())
    assert indices[0] == 0
    assert indices == list(range(len(indices)))


def test_trim_default_output_path(tmp_path):
    """trim with output_path=None generates *_trimped.avi alongside input."""
    # Copy fixture to tmp so the default output goes there
    import shutil

    src = STITCH_DATA_DIR / "video1.avi"
    src_csv = STITCH_DATA_DIR / "video1.csv"
    dst = tmp_path / "video1.avi"
    dst_csv = tmp_path / "video1.csv"
    shutil.copy(src, dst)
    shutil.copy(src_csv, dst_csv)

    out = trim(dst, output_path=None, end=40)
    assert out.name == "video1_trimmed.avi"
    assert out.parent == tmp_path

    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_trim_invalid_range(tmp_path):
    """trim raises ValueError for invalid trim ranges."""
    video = STITCH_DATA_DIR / "video1.avi"

    with pytest.raises(ValueError, match="start must be >= 0"):
        trim(video, output_path=tmp_path / "out.avi", start=-1)

    with pytest.raises(ValueError, match="end must be >= 0"):
        trim(video, output_path=tmp_path / "out.avi", end=-1)

    with pytest.raises(ValueError, match="must be < total_frames"):
        trim(video, output_path=tmp_path / "out.avi", start=25, end=25)


def test_edge_scoring_selects_less_sharp():
    """Sobel edge scoring picks the less sharp frame (higher score = less gradient)."""
    uniform = np.ones((50, 50), dtype=np.uint8) * 128
    edgy = np.zeros((50, 50), dtype=np.uint8)
    edgy[:, 25:] = 255
    assert _score_edges(uniform) > _score_edges(edgy)

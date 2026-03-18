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
from mio.models.stitch import DebugRecord, FrameInfo
from mio.process.stitch import (
    CandidateFrame,
    RecordingData,
    RecordingDataBundle,
    select_best_candidate,
    score_edges,
)
from mio.process.video import crop_run
from mio.utils import hash_video, validate_video_metadata_match

STITCH_DATA_DIR = Path(__file__).parent.parent / "data" / "stitch"

EXPECTED_STITCHED_VIDEO_HASH = "245caa04878a1288c5d2915680259e1a5c37aef819d1767a0b357587ccb3d703"
EXPECTED_DEBUG_VIDEO_HASH = "e9fd27bbf7c2ab658dbe7a63206b9f370eeb561b9c2574143e168f5f40120e03"
EXPECTED_STITCHED_FRAME_COUNT = 54
EXPECTED_DEBUG_ROWS = 4
EXPECTED_CROP_VIDEO_HASH = "432642b1528fcd9ad553cfb3cc3862bef931301bd11d44dc3c2372fc379fa629"
EXPECTED_CROP_FRAME_COUNT = 30


@pytest.fixture(scope="module")
def stitch_result(tmp_path_factory):
    """Run stitch once on the trimmed fixtures, return paths to all outputs."""
    tmp = tmp_path_factory.mktemp("stitch_regression")
    debug_dir = tmp / "debug"
    debug_dir.mkdir()

    recordings = [
        RecordingData(
            video_path=STITCH_DATA_DIR / "video1.avi",
            csv_path=STITCH_DATA_DIR / "video1.csv",
        ),
        RecordingData(
            video_path=STITCH_DATA_DIR / "video2.avi",
            csv_path=STITCH_DATA_DIR / "video2.csv",
        ),
    ]

    stitched_video = tmp / "stitched.avi"
    stitched_csv = tmp / "stitched.csv"
    debug_video = debug_dir / "debug.avi"
    debug_csv = debug_dir / "debug.csv"

    bundle = RecordingDataBundle(
        recordings=recordings,
        stitched_video_writer=VideoWriter(path=stitched_video, fps=20),
        debug_video_writer=VideoWriter(path=debug_video, fps=20),
        combined_csv_path=stitched_csv,
        debug_csv_path=debug_csv,
    )
    bundle.stitch_recordings()

    return {
        "stitched_video": stitched_video,
        "stitched_csv": stitched_csv,
        "debug_video": debug_video,
        "debug_csv": debug_csv,
    }


def test_stitched_video_hash(stitch_result):
    """Stitched video content matches expected hash (frame-by-frame blake2s)."""
    assert hash_video(stitch_result["stitched_video"]) == EXPECTED_STITCHED_VIDEO_HASH


def test_stitched_video_frame_count(stitch_result):
    """Stitched video has the expected number of frames (union of frame_nums)."""
    cap = cv2.VideoCapture(str(stitch_result["stitched_video"]))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == EXPECTED_STITCHED_FRAME_COUNT


def test_stitched_csv_valid(stitch_result):
    """Stitched CSV passes video-metadata validation."""
    validate_video_metadata_match(stitch_result["stitched_video"])


def test_stitched_csv_contiguous_index(stitch_result):
    """reconstructed_frame_index in stitched CSV is contiguous 0..N-1."""
    df = pd.read_csv(stitch_result["stitched_csv"])
    indices = sorted(df["reconstructed_frame_index"].unique())
    assert indices == list(range(len(indices)))


def test_stitched_csv_frame_num_range(stitch_result):
    """Stitched CSV covers the full union of frame_nums from both inputs."""
    df = pd.read_csv(stitch_result["stitched_csv"])
    frame_nums = sorted(df["frame_num"].unique())
    assert frame_nums[0] == 2145
    assert frame_nums[-1] == 2566
    assert len(frame_nums) == EXPECTED_STITCHED_FRAME_COUNT


def test_debug_video_hash(stitch_result):
    """Debug video content matches expected hash."""
    assert hash_video(stitch_result["debug_video"]) == EXPECTED_DEBUG_VIDEO_HASH


def test_debug_csv_columns(stitch_result):
    """Debug CSV has the columns defined by DebugRecord."""
    df = pd.read_csv(stitch_result["debug_csv"])
    assert list(df.columns) == DebugRecord.header()


def test_debug_csv_row_count(stitch_result):
    """Debug CSV has one row per frame with pixel differences."""
    df = pd.read_csv(stitch_result["debug_csv"])
    assert len(df) == EXPECTED_DEBUG_ROWS


def test_debug_csv_metadata_win(stitch_result):
    """Frame 2530: video1 has 7 buffers, video2 has 8 -> metadata picks video2 (no tie)."""
    df = pd.read_csv(stitch_result["debug_csv"])
    row = df[df["frame_num"] == 2530].iloc[0]
    assert row["metadata_tie"] == False  # noqa: E712
    assert row["selected_video"] == "video2.avi"
    assert row["selected_num_buffers"] == 8
    assert row["compare_num_buffers"] == 7


def test_debug_csv_edge_scoring_tiebreaker(stitch_result):
    """Tied frames use edge scoring: selected frame has higher score (less sharp)."""
    df = pd.read_csv(stitch_result["debug_csv"])
    tied = df[df["metadata_tie"] == True]  # noqa: E712
    assert len(tied) == 3
    for _, row in tied.iterrows():
        assert row["selected_edge_score"] > row["compare_edge_score"]


def test_stitch_without_debug(tmp_path):
    """Stitch produces correct output when debug writers are None."""
    recordings = [
        RecordingData(
            video_path=STITCH_DATA_DIR / "video1.avi",
            csv_path=STITCH_DATA_DIR / "video1.csv",
        ),
        RecordingData(
            video_path=STITCH_DATA_DIR / "video2.avi",
            csv_path=STITCH_DATA_DIR / "video2.csv",
        ),
    ]

    stitched_video = tmp_path / "stitched.avi"
    stitched_csv = tmp_path / "stitched.csv"

    bundle = RecordingDataBundle(
        recordings=recordings,
        stitched_video_writer=VideoWriter(path=stitched_video, fps=20),
        debug_video_writer=None,
        combined_csv_path=stitched_csv,
        debug_csv_path=None,
    )
    bundle.stitch_recordings()

    assert hash_video(stitched_video) == EXPECTED_STITCHED_VIDEO_HASH
    validate_video_metadata_match(stitched_video)


def test_stitch_single_recording(tmp_path):
    """Single recording passes through without comparison."""
    recordings = [
        RecordingData(
            video_path=STITCH_DATA_DIR / "video1.avi",
            csv_path=STITCH_DATA_DIR / "video1.csv",
        ),
    ]

    stitched_video = tmp_path / "stitched.avi"
    stitched_csv = tmp_path / "stitched.csv"
    debug_csv = tmp_path / "debug.csv"

    bundle = RecordingDataBundle(
        recordings=recordings,
        stitched_video_writer=VideoWriter(path=stitched_video, fps=20),
        combined_csv_path=stitched_csv,
        debug_csv_path=debug_csv,
    )
    bundle.stitch_recordings()

    # video1 has 50 video frames but only 49 unique frame_nums.
    # frame_num 2535 appears at rfi 22 (8 buffers) and rfi 31 (1 buffer).
    # The stitcher deduplicates by frame_num, using majority rfi (22).
    # So the output has 49 frames, not 50.
    cap = cv2.VideoCapture(str(stitched_video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 49
    cap.release()

    validate_video_metadata_match(stitched_video)

    # frame_num 2535 appears once in output (deduplicated), but all 9 metadata rows
    # are included (8 from rfi=22 + 1 fragment from rfi=31)
    df_out = pd.read_csv(stitched_csv)
    rows_2535 = df_out[df_out["frame_num"] == 2535]
    assert len(rows_2535) == 9

    # No debug rows (nothing to compare)
    df_debug = pd.read_csv(debug_csv)
    assert len(df_debug) == 0


def test_stitch_padding_tiebreaker(tmp_path):
    """When buffer counts are equal, less black_padding_px wins.

    Uses video2_padded.csv where frame 2520 has black_padding_px=100.
    Both recordings have 8 buffers for this frame, so buffer count ties.
    video1 (padding=0) should be selected over video2 (padding=100).
    """
    recordings = [
        RecordingData(
            video_path=STITCH_DATA_DIR / "video1.avi",
            csv_path=STITCH_DATA_DIR / "video1.csv",
        ),
        RecordingData(
            video_path=STITCH_DATA_DIR / "video2.avi",
            csv_path=STITCH_DATA_DIR / "video2_padded.csv",
        ),
    ]

    stitched_video = tmp_path / "stitched.avi"
    stitched_csv = tmp_path / "stitched.csv"

    bundle = RecordingDataBundle(
        recordings=recordings,
        stitched_video_writer=VideoWriter(path=stitched_video, fps=20),
        combined_csv_path=stitched_csv,
    )
    bundle.stitch_recordings()

    # Verify video1 was selected for frame 2520 (less padding)
    df_out = pd.read_csv(stitched_csv)
    rows = df_out[df_out["frame_num"] == 2520]
    assert len(rows) > 0
    assert all(rows["black_padding_px"] == 0)


def test_crop_video_hash(tmp_path):
    """Cropped video content matches expected hash."""
    out = crop_run(
        str(STITCH_DATA_DIR / "video1.avi"),
        output_path=str(tmp_path / "cropped.avi"),
        trim_start=10,
        trim_end=10,
    )
    assert hash_video(out) == EXPECTED_CROP_VIDEO_HASH


def test_crop_frame_count(tmp_path):
    """Cropped video has the expected number of frames."""
    out = crop_run(
        str(STITCH_DATA_DIR / "video1.avi"),
        output_path=str(tmp_path / "cropped.avi"),
        trim_start=10,
        trim_end=10,
    )
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == EXPECTED_CROP_FRAME_COUNT
    cap.release()


def test_crop_csv_valid(tmp_path):
    """Cropped CSV passes video-metadata validation."""
    out = crop_run(
        str(STITCH_DATA_DIR / "video1.avi"),
        output_path=str(tmp_path / "cropped.avi"),
        trim_start=10,
        trim_end=10,
    )
    validate_video_metadata_match(out)


def test_crop_csv_renumbered(tmp_path):
    """Cropped CSV has reconstructed_frame_index renumbered to 0-based."""
    out = crop_run(
        str(STITCH_DATA_DIR / "video1.avi"),
        output_path=str(tmp_path / "cropped.avi"),
        trim_start=10,
        trim_end=10,
    )
    df = pd.read_csv(str(out).replace(".avi", ".csv"))
    indices = sorted(df["reconstructed_frame_index"].unique())
    assert indices[0] == 0
    assert indices == list(range(len(indices)))


def test_crop_default_output_path(tmp_path):
    """crop_run with output_path=None generates *_cropped.avi alongside input."""
    # Copy fixture to tmp so the default output goes there
    import shutil

    src = STITCH_DATA_DIR / "video1.avi"
    src_csv = STITCH_DATA_DIR / "video1.csv"
    dst = tmp_path / "video1.avi"
    dst_csv = tmp_path / "video1.csv"
    shutil.copy(src, dst)
    shutil.copy(src_csv, dst_csv)

    out = crop_run(str(dst), output_path=None, trim_end=40)
    assert out.name == "video1_cropped.avi"
    assert out.parent == tmp_path

    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_crop_invalid_range(tmp_path):
    """crop_run raises ValueError for invalid trim ranges."""
    video = str(STITCH_DATA_DIR / "video1.avi")

    with pytest.raises(ValueError, match="trim_start must be >= 0"):
        crop_run(video, output_path=str(tmp_path / "out.avi"), trim_start=-1)

    with pytest.raises(ValueError, match="trim_end must be >= 0"):
        crop_run(video, output_path=str(tmp_path / "out.avi"), trim_end=-1)

    with pytest.raises(ValueError, match="must be < total_frames"):
        crop_run(video, output_path=str(tmp_path / "out.avi"), trim_start=25, trim_end=25)


def test_metadata_tie_detection():
    """Equal metadata scores are detected as a tie (triggers edge scoring)."""
    cand = CandidateFrame(
        recording=None,
        frame=None,
        num_buffers=8,
        sum_black_padding=0,
        metadata_rows=None,
        _edge_score=0.0,
    )
    _, is_tie = select_best_candidate([cand, cand])
    assert is_tie is True


def test_edge_scoring_selects_less_sharp():
    """Sobel edge scoring picks the less sharp frame (higher score = less gradient)."""
    uniform = np.ones((50, 50), dtype=np.uint8) * 128
    edgy = np.zeros((50, 50), dtype=np.uint8)
    edgy[:, 25:] = 255
    assert score_edges(uniform) > score_edges(edgy)


def test_frame_info_majority_vote_rfi():
    """When a frame_num maps to multiple rfi values, majority wins."""
    base = {
        "frame_num": 200,
        "buffer_count": 1,
        "frame_buffer_count": 8,
        "timestamp": 2000,
        "pixel_count": 5000,
        "black_padding_px": 0,
        "buffer_recv_unix_time": 2.0,
    }
    rows = [{**base, "reconstructed_frame_index": 10, "buffer_recv_index": i} for i in range(7)] + [
        {**base, "reconstructed_frame_index": 20, "buffer_recv_index": 7}
    ]
    fi = FrameInfo.from_metadata(200, pd.DataFrame(rows))
    assert fi.reconstructed_frame_index == 10

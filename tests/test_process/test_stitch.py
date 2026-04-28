"""
Regression tests for the stitch feature.

Fixtures are trimmed from real recordings.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from mio.devices.tables import StitchRecord
from mio.models.dataset import Recording, StitchedRecording
from mio.process.stitch import (
    _align_by_time,
    _has_discontinuous_runs,
    _score_edges,
    concat_recordings,
    stitch,
)
from mio.process.video import remove_frames, trim
from mio.utils import hash_video

STITCH_DATA_DIR = Path(__file__).parent.parent / "data" / "stitch"

EXPECTED_STITCHED_VIDEO_HASH = "df937c8651cf142b4d8e2a75140729dcacdc1151ebc3767b48d0ca71578007ff"
EXPECTED_DEBUG_VIDEO_HASH = (
    "856e6e5c538532bd0fcfb942616686a5cd262aadb51dd8796adf5de69215c94b",
    "a69b6cadf4ab1dd8a1097d2c1be298397206db235fd4c5f68febd1700f15a4b6",
)
EXPECTED_CROP_VIDEO_HASH = "432642b1528fcd9ad553cfb3cc3862bef931301bd11d44dc3c2372fc379fa629"
EXPECTED_CROP_FRAME_COUNT = 30
EXPECTED_VIDEO1_FRAME_COUNT = 50
EXPECTED_REMOVE_FRAMES_HASH = "b76b80f45316bad0a808802b8f5c0d65b99f6f59bc6422b84c1c2a7026ca4b15"


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
    assert hash_video(stitch_result.debug_video.path) in EXPECTED_DEBUG_VIDEO_HASH


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
    # there should be 7 frames that could be decided on metadata alone
    # - 4x on buffer count
    # - 3x on black pixels
    assert len(df[df["selected_edge_score"].isna()]) == 7
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


def test_remove_frames(tmp_path):
    """End-to-end: remove specific frames, verify video hash, frame count, and CSV integrity."""
    out = remove_frames(
        STITCH_DATA_DIR / "video1.avi",
        remove_indices=[0, 5, 10],
        output_path=tmp_path / "removed.avi",
    )

    assert hash_video(out.video.path) == EXPECTED_REMOVE_FRAMES_HASH

    assert int(out.video.get(cv2.CAP_PROP_FRAME_COUNT)) == EXPECTED_VIDEO1_FRAME_COUNT - 3

    df = out.metadata
    indices = sorted(df["reconstructed_frame_index"].unique())
    assert indices == list(range(len(indices)))


def test_remove_frames_invalid(tmp_path):
    """Invalid frame indices are rejected before processing."""
    video = STITCH_DATA_DIR / "video1.avi"

    with pytest.raises(ValueError, match="out of range"):
        remove_frames(video, remove_indices=[-1], output_path=tmp_path / "out.avi")

    with pytest.raises(ValueError, match="out of range"):
        remove_frames(video, remove_indices=[9999], output_path=tmp_path / "out.avi")

    with pytest.raises(ValueError, match="Cannot remove all"):
        remove_frames(
            video,
            remove_indices=list(range(EXPECTED_VIDEO1_FRAME_COUNT)),
            output_path=tmp_path / "out.avi",
        )


def test_concat_recordings(tmp_path, recordings):
    """Concatenating two recordings produces contiguous frame indices and correct frame count."""

    combined_video = tmp_path / "combined.avi"

    combined = concat_recordings(
        recordings=list(recordings.values()),
        output_video_path=combined_video,
    )

    # Video frame count should be sum of both inputs
    expected_frames = sum(r.video.n_frames for r in recordings.values())
    # the Recording class also validates that the metadata has a matching length if present
    # so its presence means that the metadata is also matching
    assert combined.metadata is not None

    actual_frames = combined.video.n_frames
    assert actual_frames == expected_frames

    # CSV should have contiguous reconstructed_frame_index
    df = combined.metadata
    diffs = df["reconstructed_frame_index"].diff().iloc[1:].to_numpy()
    assert (diffs <= 1).all() and (diffs >= 0).all()


@pytest.mark.parametrize("flip", [True, False])
def test_align_by_time(tmp_path, flip):
    """Aligning by timestamp finds the inner join of the closest matching timestamps"""
    # one normal video with some linspaced times
    left_idxes = np.ravel(np.repeat(np.arange(50), 5))
    left_times = np.linspace(0, 1, len(left_idxes))
    left = pd.DataFrame(
        {"reconstructed_frame_index": left_idxes, "buffer_recv_unix_time": left_times}
    )

    # one offset video with a blippy frame from a bit flip in the frame_num
    right_idxes = np.ravel(np.repeat(np.arange(25), 5))
    right_idxes = np.concat([right_idxes, [25]], axis=0)
    right_idxes = np.concat([right_idxes, np.ravel(np.repeat(np.arange(26, 52), 5))], axis=0)
    # make same size so sampling rate is the same
    right_idxes = right_idxes[: len(left_idxes)]
    right_times = np.linspace(0, 1, len(right_idxes)) + 0.1
    right = pd.DataFrame(
        {"reconstructed_frame_index": right_idxes, "buffer_recv_unix_time": right_times}
    )

    good, bad = "video1", "video2"
    if flip:
        good, bad = "video2", "video1"
    recordings = [
        Recording.model_construct(name=good, metadata=left),
        Recording.model_construct(name=bad, metadata=right),
    ]

    aligned = _align_by_time(recordings)
    # we should have received 45 frames: 50 frames in the original - 5 frames in 0.1 seconds of lag
    assert len(aligned) == 45
    assert np.array_equal(aligned[good], np.arange(5, 50))

    # we should have dropped frame 25 in the right one
    assert 25 not in np.array(aligned[bad])
    assert np.array_equal(aligned[bad], np.concat([np.arange(25), np.arange(26, 46)]))


def test_stitch_with_timestamps(stitch_result, tmp_path):
    """
    When we scramble the `frame_num`, we can stitch by timestamps.
    We should get the same result as if we were able to use frame_num in this case.
    """
    # use a temporary version of the recordings because we are going to wreck the metadata
    recordings = {
        "video1": Recording.from_video(STITCH_DATA_DIR / "video1.avi"),
        "video2": Recording.from_video(STITCH_DATA_DIR / "video2.avi"),
    }
    recordings["video1"].metadata["frame_num"] = np.random.default_rng().integers(
        0, 1000, size=len(recordings["video1"].metadata)
    )
    recordings["video2"].metadata["frame_num"] = np.random.default_rng().integers(
        0, 1000, size=len(recordings["video2"].metadata)
    )

    result = stitch(list(recordings.values()), debug_video=True, output_dir=tmp_path)
    # we should have an inner join on the frames - so only those without a comparison frame
    expected = stitch_result.scores[~stitch_result.scores["compare_video"].isna()]
    assert np.array_equal(result.scores["selected_video"], expected["selected_video"])


@pytest.mark.parametrize(
    "series,expected",
    [
        pytest.param([1, 1, 1, 2, 2, 2, 3, 3, 3], False, id="contiguous-buffers"),
        pytest.param([1, 2, 3, 4, 5], False, id="contiguous-frames"),
        pytest.param([1, 1, 1, 2, 500, 2, 3, 3, 3], False, id="single-bitflip-same-frame"),
        pytest.param([1, 1, 1, 2, 2, 500, 3, 3, 3], False, id="single-bitflip-next-frame"),
        pytest.param(
            [10, 10, 10, 11, 11, 11, 2, 2, 2, 3, 3, 3], True, id="discontiguous-buffers-lower"
        ),
        pytest.param(
            [10, 10, 10, 11, 11, 11, 20, 20, 20, 21, 21, 21],
            True,
            id="discontiguous-buffers-higher",
        ),
        pytest.param([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], True, id="discontiguous-frames-lower"),
        pytest.param([1, 2, 3, 4, 5, 10, 11, 12, 13], True, id="discontiguous-frames-higher"),
    ],
)
def test_has_discontinuous_runs(series, expected):
    """
    We can determine when some timeseries has discontinuous runs,
    ignoring when a single value blips incorrectly (like a bit flip in a metadata header).
    """
    series = pd.Series(series)
    assert _has_discontinuous_runs(series) == expected


def test_test_data_is_considered_continuous(recordings):
    """Just testing the assumptions of the tests ios all"""
    assert not any(_has_discontinuous_runs(r.metadata["frame_num"]) for r in recordings.values())

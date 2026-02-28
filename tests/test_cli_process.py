"""
CLI tests for the process commands (stitch, crop).

Uses Click's CliRunner and the same trimmed fixtures as the stitch regression tests.
"""

import shutil
from pathlib import Path

from click.testing import CliRunner

from mio.cli.process import process
from mio.utils import hash_video, validate_video_metadata_match

STITCH_DATA_DIR = Path(__file__).parent / "data" / "stitch"

EXPECTED_STITCHED_VIDEO_HASH = (
    "245caa04878a1288c5d2915680259e1a5c37aef819d1767a0b357587ccb3d703"
)
EXPECTED_CROP_VIDEO_HASH = (
    "432642b1528fcd9ad553cfb3cc3862bef931301bd11d44dc3c2372fc379fa629"
)


def test_cli_stitch(tmp_path):
    """mio process stitch produces correct stitched output."""
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "stitch",
            "-i", str(STITCH_DATA_DIR / "video1.avi"),
            "-i", str(STITCH_DATA_DIR / "video2.avi"),
            "-o", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    out_video = out_dir / "video1_stitched.avi"
    assert hash_video(out_video) == EXPECTED_STITCHED_VIDEO_HASH
    is_valid, msg, _ = validate_video_metadata_match(out_video)
    assert is_valid, msg


def test_cli_stitch_with_debug(tmp_path):
    """mio process stitch with debug flags produces debug outputs."""
    out_dir = tmp_path / "out"
    debug_video = tmp_path / "debug.avi"
    debug_csv = tmp_path / "debug.csv"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "stitch",
            "-i", str(STITCH_DATA_DIR / "video1.avi"),
            "-i", str(STITCH_DATA_DIR / "video2.avi"),
            "-o", str(out_dir),
            "--debug-video", str(debug_video),
            "--debug-csv", str(debug_csv),
        ],
    )
    assert result.exit_code == 0, result.output
    assert debug_video.exists()
    assert debug_csv.exists()


def test_cli_stitch_single_input_rejected(tmp_path):
    """mio process stitch rejects a single input."""
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "stitch",
            "-i", str(STITCH_DATA_DIR / "video1.avi"),
            "-o", str(tmp_path / "out.avi"),
        ],
    )
    assert result.exit_code != 0
    assert "At least 2" in result.output


def test_cli_crop(tmp_path):
    """mio process crop produces correct cropped output."""
    # Copy fixture to tmp so default CSV lookup works
    src = STITCH_DATA_DIR / "video1.avi"
    src_csv = STITCH_DATA_DIR / "video1.csv"
    dst = tmp_path / "video1.avi"
    dst_csv = tmp_path / "video1.csv"
    shutil.copy(src, dst)
    shutil.copy(src_csv, dst_csv)

    out_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "crop",
            "-i", str(dst),
            "-o", str(out_dir),
            "-s", "10",
            "-e", "39",
        ],
    )
    assert result.exit_code == 0, result.output
    out_video = out_dir / "video1_cropped.avi"
    assert out_video.exists()
    assert hash_video(out_video) == EXPECTED_CROP_VIDEO_HASH


def test_cli_crop_no_trim(tmp_path):
    """mio process crop with no trim flags copies entire video."""
    src = STITCH_DATA_DIR / "video1.avi"
    src_csv = STITCH_DATA_DIR / "video1.csv"
    dst = tmp_path / "video1.avi"
    dst_csv = tmp_path / "video1.csv"
    shutil.copy(src, dst)
    shutil.copy(src_csv, dst_csv)

    out_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "crop",
            "-i", str(dst),
            "-o", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No trimming" in result.output

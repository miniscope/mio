"""
CLI tests for the process commands (stitch, trim).

Uses Click's CliRunner and the same trimmed fixtures as the stitch regression tests.
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from mio.cli.process import process
from mio.exceptions import VideoMetadataError
from mio.utils import hash_video

STITCH_DATA_DIR = Path(__file__).parent / "data" / "stitch"

EXPECTED_STITCHED_VIDEO_HASH = "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"
EXPECTED_CROP_VIDEO_HASH = "432642b1528fcd9ad553cfb3cc3862bef931301bd11d44dc3c2372fc379fa629"


def test_cli_stitch(tmp_path):
    """mio process stitch produces correct stitched output."""
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "stitch",
            "-i",
            str(STITCH_DATA_DIR / "video1.avi"),
            "-i",
            str(STITCH_DATA_DIR / "video2.avi"),
            "-o",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    out_video = out_dir / "video1_stitched.avi"
    assert hash_video(out_video) == EXPECTED_STITCHED_VIDEO_HASH


def test_cli_stitch_with_debug(tmp_path):
    """mio process stitch with debug flags produces debug outputs."""
    out_dir = tmp_path / "out"
    debug_video = out_dir / "video1__video2_stitched_debug.avi"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "stitch",
            "-i",
            str(STITCH_DATA_DIR / "video1.avi"),
            "-i",
            str(STITCH_DATA_DIR / "video2.avi"),
            "-o",
            str(out_dir),
            "--debug-video",
        ],
    )
    assert result.exit_code == 0, result.output
    assert debug_video.exists()


def test_cli_stitch_single_input_rejected(tmp_path):
    """mio process stitch rejects a single input."""
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "stitch",
            "-i",
            str(STITCH_DATA_DIR / "video1.avi"),
            "-o",
            str(tmp_path / "out.avi"),
        ],
    )
    assert result.exit_code != 0
    assert "At least 2" in result.output


def test_cli_trim(tmp_path):
    """mio process trim produces correct cropped output."""
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
            "trim",
            "-i",
            str(dst),
            "-o",
            str(out_dir),
            "-s",
            "10",
            "-e",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    out_video = out_dir / "video1_trimmed.avi"
    assert out_video.exists()
    assert hash_video(out_video) == EXPECTED_CROP_VIDEO_HASH


def test_cli_trim_no_trim(tmp_path):
    """mio process trim with no trim flags copies entire video."""
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
            "trim",
            "-i",
            str(dst),
            "-o",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

"""
CLI tests for the process commands (stitch, trim).

Uses Click's CliRunner and the same trimmed fixtures as the stitch regression tests.
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from mio.cli.process import process
from mio.utils import hash_video

STITCH_DATA_DIR = Path(__file__).parent / "data" / "stitch"

EXPECTED_STITCHED_VIDEO_HASH = "c8cdf3149f812ae25e6f3f1a876249e4ce118e9a53aa1805e48b995b01f07a91"
EXPECTED_CROP_VIDEO_HASH = "432642b1528fcd9ad553cfb3cc3862bef931301bd11d44dc3c2372fc379fa629"
EXPECTED_STITCHED_TRIMMED_VIDEO_HASH = (
    "2c62b65ddd537e94e7d3f29e7c46523357d70aefed02d46baa9726ee57798af9"
)


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
    out_video = out_dir / "video1__video2_stitched.avi"
    assert out_video.exists()
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


def test_cli_trim_stitched(tmp_path):
    """mio process trim on a stitched output produces correct cropped video."""
    out_dir = tmp_path / "out"
    runner = CliRunner()
    stitch_result = runner.invoke(
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
    assert stitch_result.exit_code == 0, stitch_result.output
    stitched = out_dir / "video1__video2_stitched.avi"
    trim_result = runner.invoke(
        process,
        [
            "trim",
            "-i",
            str(stitched),
            "-o",
            str(out_dir),
            "-s",
            "2",
            "-e",
            "2",
        ],
    )
    assert trim_result.exit_code == 0, trim_result.output
    trimmed = out_dir / "video1__video2_stitched_trimmed.avi"
    assert trimmed.exists()
    assert hash_video(trimmed) == EXPECTED_STITCHED_TRIMMED_VIDEO_HASH


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


def test_cli_workflow(tmp_path):
    """mio process workflow (stitch + denoise) completes end-to-end."""
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "workflow",
            "-i",
            str(STITCH_DATA_DIR / "video1.avi"),
            "-i",
            str(STITCH_DATA_DIR / "video2.avi"),
            "-o",
            str(out_dir),
            "-c",
            "denoise_example",
        ],
    )
    assert result.exit_code == 0, result.output
    stitched = out_dir / "video1__video2_stitched.avi"
    denoised = out_dir / "user_data" / "output" / "video1__video2_stitched_patch.avi"
    assert stitched.exists()
    assert denoised.exists()
    assert hash_video(stitched) == EXPECTED_STITCHED_VIDEO_HASH


def test_cli_workflow_with_trim(tmp_path):
    """mio process workflow with trim completes end-to-end on stitched input."""
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        process,
        [
            "workflow",
            "-i",
            str(STITCH_DATA_DIR / "video1.avi"),
            "-i",
            str(STITCH_DATA_DIR / "video2.avi"),
            "-o",
            str(out_dir),
            "-c",
            "denoise_example",
            "--trim-start",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    trimmed = out_dir / "video1__video2_stitched_trimmed.avi"
    denoised = out_dir / "user_data" / "output" / "video1__video2_stitched_trimmed_patch.avi"
    assert trimmed.exists()
    assert denoised.exists()

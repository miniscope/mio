"""
Command line interface for offline video pre-processing.
"""

from pathlib import Path
from typing import List, Optional

import click

from mio.io import VideoWriter
from mio.models.process import DenoiseConfig
from mio.process.stitch import RecordingData, RecordingDataBundle
from mio.process.video import crop_run, denoise_run
from mio.utils import validate_frame_count_alignment, validate_video_metadata_match


@click.group()
def process() -> None:
    """
    Command group for video processing.
    """
    pass


@process.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the video file to process.",
)
@click.option(
    "-c",
    "--denoise_config",
    required=True,
    type=str,
    help="Path to the YAML processing configuration file.",
)
def denoise(
    input: str,
    denoise_config: str,
) -> None:
    """
    Denoise a video file.
    """
    # Validate video/metadata match at the beginning
    is_valid, error_msg, csv_df = validate_video_metadata_match(input)

    if not is_valid:
        if error_msg and "not found" in error_msg.lower():
            if click.confirm(
                f"{error_msg}. Do you want to continue without generating output CSV metadata?",
                default=False,
            ):
                click.echo(f"Warning: {error_msg}. Continuing without CSV metadata generation.")
            else:
                raise click.ClickException(f"{error_msg}. Cannot proceed without CSV.")
        else:
            if click.confirm(
                f"{error_msg}. This may indicate a mismatch between the video and CSV. "
                "Do you want to continue anyway?",
                default=False,
            ):
                click.echo(f"Warning: {error_msg}. Proceeding anyway.")
            else:
                raise click.ClickException(f"{error_msg}. Cannot proceed.")

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)
    denoise_run(
        input,
        denoise_config_parsed,
        csv_validation_result=(is_valid, csv_df),
    )


@process.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the video file to crop.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to the output video file. If not specified, "
    "defaults to input path with '_cropped' suffix.",
)
@click.option(
    "-s",
    "--trim-start",
    type=int,
    default=None,
    help="Start frame index for cropping (0-based, inclusive).",
)
@click.option(
    "-e",
    "--trim-end",
    type=int,
    default=None,
    help="End frame index for cropping (0-based, inclusive).",
)
def crop(
    input: str,
    output: Optional[str],
    trim_start: Optional[int],
    trim_end: Optional[int],
) -> None:
    """
    Crop a video file by trimming frames.
    """
    if trim_start is None and trim_end is None:
        raise click.ClickException(
            "At least one of --trim-start or --trim-end must be specified."
        )

    # Validate video/metadata match at the beginning
    is_valid, error_msg, csv_df = validate_video_metadata_match(input)

    if not is_valid:
        if error_msg and "not found" in error_msg.lower():
            if click.confirm(
                f"{error_msg}. Do you want to continue without generating "
                "output CSV metadata?",
                default=False,
            ):
                click.echo(
                    f"Warning: {error_msg}. "
                    "Continuing without CSV metadata generation."
                )
            else:
                raise click.ClickException(
                    f"{error_msg}. Cannot proceed without CSV."
                )
        else:
            if click.confirm(
                f"{error_msg}. This may indicate a mismatch between the "
                "video and CSV. Do you want to continue anyway?",
                default=False,
            ):
                click.echo(f"Warning: {error_msg}. Proceeding anyway.")
            else:
                raise click.ClickException(f"{error_msg}. Cannot proceed.")

    crop_run(
        input,
        output_path=output,
        csv_validation_result=(is_valid, csv_df),
        trim_start=trim_start,
        trim_end=trim_end,
    )

    # Validate frame count alignment after cropping
    if output is not None:
        output_path_obj = Path(output)
    else:
        input_path = Path(input)
        output_path_obj = input_path.parent / f"{input_path.stem}_cropped{input_path.suffix}"

    is_aligned, alignment_error = validate_frame_count_alignment(output_path_obj)
    if not is_aligned:
        raise click.ClickException(
            f"Frame count alignment failed after cropping: {alignment_error}"
        )
    click.echo(f"✓ Frame count alignment verified: {output_path_obj}")


@process.command()
@click.option(
    "-i",
    "--inputs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Paths to video files to stitch (can be specified multiple times).",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to the output stitched video file.",
)
@click.option(
    "--debug-video",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to optional debug video file.",
)
@click.option(
    "--debug-csv",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to optional debug CSV file.",
)
@click.option(
    "--fps",
    type=int,
    default=20,
    help="Frames per second for output video.",
)
def stitch(
    inputs: tuple,
    output: str,
    debug_video: Optional[str],
    debug_csv: Optional[str],
    fps: int,
) -> None:
    """
    Stitch multiple video recordings together.
    """
    if len(inputs) < 2:
        raise click.ClickException("At least 2 input videos are required for stitching.")

    # Create RecordingData objects
    recordings: List[RecordingData] = []
    for video_path in inputs:
        video_path_obj = Path(video_path)
        csv_path_obj = video_path_obj.with_suffix(".csv")
        if not csv_path_obj.exists():
            raise click.ClickException(
                f"CSV file not found for {video_path}: {csv_path_obj}"
            )
        recordings.append(RecordingData(video_path=video_path_obj, csv_path=csv_path_obj))

    # Create output paths
    output_path_obj = Path(output)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_path_obj.with_suffix(".csv")

    debug_video_path = Path(debug_video) if debug_video else None
    debug_csv_path = Path(debug_csv) if debug_csv else None

    # Create video writers
    combined_video_writer = VideoWriter(path=output_path_obj, fps=fps)
    debug_video_writer = (
        VideoWriter(path=debug_video_path, fps=fps) if debug_video_path else None
    )

    # Create bundle and stitch
    recording_bundle = RecordingDataBundle(
        recordings=recordings,
        combined_video_writer=combined_video_writer,
        debug_video_writer=debug_video_writer,
        combined_csv_path=output_csv_path,
        debug_csv_path=debug_csv_path,
    )

    click.echo(f"Stitching {len(recordings)} recordings...")
    recording_bundle.stitch_recordings()

    # Validate frame count alignment after stitching
    is_aligned, alignment_error = validate_frame_count_alignment(output_path_obj)
    if not is_aligned:
        raise click.ClickException(
            f"Frame count alignment failed after stitching: {alignment_error}"
        )
    click.echo(f"✓ Frame count alignment verified: {output_path_obj}")


@process.command()
@click.option(
    "-i",
    "--inputs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Paths to video files to stitch (can be specified multiple times).",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Base path for output files (stitched, cropped, denoised).",
)
@click.option(
    "-c",
    "--denoise_config",
    required=True,
    type=str,
    help="Path to the YAML processing configuration file.",
)
@click.option(
    "--trim-start",
    type=int,
    default=1200,
    help="Number of frames to remove from the start (default: 1200).",
)
@click.option(
    "--trim-end",
    type=int,
    default=200,
    help="Number of frames to remove from the end (default: 200).",
)
@click.option(
    "--fps",
    type=int,
    default=20,
    help="Frames per second for stitched video.",
)
def workflow(
    inputs: tuple,
    output: str,
    denoise_config: str,
    trim_start: int,
    trim_end: int,
    fps: int,
) -> None:
    """
    Complete workflow: stitch → trim → denoise with validation at each step.
    """
    output_base = Path(output)
    output_dir = output_base.parent
    output_stem = output_base.stem

    import cv2

    # Step 1: Stitch
    stitched_video = output_dir / f"{output_stem}_stitched.avi"
    click.echo("=" * 60)
    click.echo("Step 1: Stitching videos...")
    click.echo("=" * 60)

    # Create RecordingData objects
    recordings: List[RecordingData] = []
    for video_path in inputs:
        video_path_obj = Path(video_path)
        csv_path_obj = video_path_obj.with_suffix(".csv")
        if not csv_path_obj.exists():
            raise click.ClickException(
                f"CSV file not found for {video_path}: {csv_path_obj}"
            )
        recordings.append(RecordingData(video_path=video_path_obj, csv_path=csv_path_obj))

    # Create output paths
    output_path_obj = Path(stitched_video)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_path_obj.with_suffix(".csv")

    # Create video writers
    combined_video_writer = VideoWriter(path=output_path_obj, fps=fps)

    # Create bundle and stitch
    recording_bundle = RecordingDataBundle(
        recordings=recordings,
        combined_video_writer=combined_video_writer,
        debug_video_writer=None,
        combined_csv_path=output_csv_path,
        debug_csv_path=None,
    )

    click.echo(f"Stitching {len(recordings)} recordings...")
    recording_bundle.stitch_recordings()

    # Validate frame count alignment after stitching
    is_aligned, alignment_error = validate_frame_count_alignment(output_path_obj)
    if not is_aligned:
        raise click.ClickException(
            f"Frame count alignment failed after stitching: {alignment_error}"
        )
    click.echo(f"✓ Frame count alignment verified: {output_path_obj}")
    click.echo(f"✓ Saved stitched video: {output_path_obj}")
    click.echo(f"✓ Saved stitched metadata: {output_csv_path}")

    # Step 2: Trim (remove first trim_start frames and last trim_end frames)
    cropped_video = output_dir / f"{output_stem}_cropped.avi"
    click.echo("=" * 60)
    click.echo("Step 2: Trimming video...")
    click.echo("=" * 60)

    # Get video frame count to calculate trim range
    cap = cv2.VideoCapture(str(stitched_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop_start = trim_start
    crop_end = total_frames - trim_end - 1  # -1 because trim_end is inclusive

    if crop_start >= crop_end:
        raise click.ClickException(
            f"Invalid trim range: start={crop_start}, end={crop_end}. "
            f"Total frames: {total_frames}"
        )

    click.echo(
        f"Trimming: removing first {trim_start} frames and last {trim_end} frames "
        f"(keeping frames {crop_start}-{crop_end})"
    )

    # Validate input before cropping
    is_valid, error_msg, csv_df = validate_video_metadata_match(str(stitched_video))
    if not is_valid:
        raise click.ClickException(f"Cannot crop: {error_msg}")

    crop_run(
        str(stitched_video),
        output_path=str(cropped_video),
        csv_validation_result=(is_valid, csv_df),
        trim_start=crop_start,
        trim_end=crop_end,
    )

    # Validate frame count alignment after cropping
    is_aligned, alignment_error = validate_frame_count_alignment(cropped_video)
    if not is_aligned:
        raise click.ClickException(
            f"Frame count alignment failed after cropping: {alignment_error}"
        )
    click.echo(f"✓ Frame count alignment verified: {cropped_video}")
    click.echo(f"✓ Saved cropped video: {cropped_video}")
    click.echo(f"✓ Saved cropped metadata: {cropped_video.with_suffix('.csv')}")

    # Step 3: Denoise
    click.echo("=" * 60)
    click.echo("Step 3: Denoising video...")
    click.echo("=" * 60)

    # Validate input before denoising
    is_valid, error_msg, csv_df = validate_video_metadata_match(str(cropped_video))
    if not is_valid:
        if error_msg and "not found" in error_msg.lower():
            raise click.ClickException(f"{error_msg}. Cannot proceed without CSV.")
        else:
            if not click.confirm(
                f"{error_msg}. This may indicate a mismatch between the video and CSV. "
                "Do you want to continue anyway?",
                default=False,
            ):
                raise click.ClickException(f"{error_msg}. Cannot proceed.")

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)
    denoise_run(
        str(cropped_video),
        denoise_config_parsed,
        csv_validation_result=(is_valid, csv_df),
    )

    # Final validation and output summary
    # The denoise output is in the config output_dir, need to find it
    output_dir_denoise = Path.cwd() / denoise_config_parsed.output_dir
    # Find the output video (it will be named output_<name>.avi)
    output_videos = list(output_dir_denoise.glob("output_*.avi"))
    if output_videos:
        final_video = output_videos[0]
        final_csv = final_video.with_suffix(".csv")
        click.echo("=" * 60)
        click.echo("Final validation...")
        click.echo("=" * 60)
        is_aligned, alignment_error = validate_frame_count_alignment(final_video)
        if not is_aligned:
            raise click.ClickException(
                f"Frame count alignment failed after denoising: {alignment_error}"
            )
        click.echo(f"✓ Final frame count alignment verified: {final_video}")
        click.echo(f"✓ Saved denoised video: {final_video}")
        if final_csv.exists():
            click.echo(f"✓ Saved denoised metadata: {final_csv}")
    else:
        click.echo("Warning: Could not find denoised output video for validation.")

    click.echo("=" * 60)
    click.echo("Workflow completed successfully!")
    click.echo("=" * 60)
    click.echo("\nOutput files:")
    click.echo(f"  1. Stitched:   {output_path_obj}")
    click.echo(f"  2. Cropped:    {cropped_video}")
    if output_videos:
        click.echo(f"  3. Denoised:   {final_video}")
    click.echo("=" * 60)

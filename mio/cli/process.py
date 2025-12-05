"""
Command line interface for offline video pre-processing.
"""

from pathlib import Path
from typing import List, Optional

import click

from mio.io import VideoWriter
from mio.logging import init_logger
from mio.models.process import DenoiseConfig
from mio.process.stitch import RecordingData, RecordingDataBundle
from mio.process.video import crop_run, denoise_run
from mio.utils import validate_frame_count_alignment, validate_video_metadata_match

logger = init_logger("mio.cli.process")

DEFAULT_PROCESS_DIR = "mio_process"


def resolve_output_path(
    input_path: Path,
    suffix: str,
    output: str = DEFAULT_PROCESS_DIR,
) -> Path:
    """
    Resolve the output path for process commands.
    Treats output as a directory and generates filename with suffix.
    
    Parameters:
    input_path: Path to the input file
    suffix: Suffix to add to filename (e.g., "_cropped", "_stitched")
    output: Output directory (default: DEFAULT_PROCESS_DIR)
    
    Returns:
    Resolved output Path

    .. todo::
        Might be better to make this a genereic helper.
        Putting it in cli for nowas we want this for process commands.
    """
    output_dir = Path(output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


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
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default=None,
    help=(
        f"Output directory for denoised files. "
        f"If not specified, uses {DEFAULT_PROCESS_DIR}/ directory."
    ),
)
def denoise(
    input: str,
    denoise_config: str,
    output_dir: Optional[str],
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
                logger.warning(f"{error_msg}. Continuing without CSV metadata generation.")
            else:
                raise click.ClickException(f"{error_msg}. Cannot proceed without CSV.")
        else:
            if click.confirm(
                f"{error_msg}. This may indicate a mismatch between the video and CSV. "
                "Do you want to continue anyway?",
                default=False,
            ):
                logger.warning(f"{error_msg}. Proceeding anyway.")
            else:
                raise click.ClickException(f"{error_msg}. Cannot proceed.")

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)
    
    # Override output_dir if not specified in config or if user provided one
    if output_dir is not None:
        denoise_config_parsed.output_dir = str(Path(output_dir).expanduser())
    elif denoise_config_parsed.output_dir is None:
        # Use organized mio_process directory
        default_output_dir = Path.cwd() / DEFAULT_PROCESS_DIR
        default_output_dir.mkdir(parents=True, exist_ok=True)
        denoise_config_parsed.output_dir = str(default_output_dir)
    
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
    type=click.Path(),
    default=None,
    help="Path to the output video file or directory. If a directory, "
    "the output filename will be generated from the input filename. "
    f"If not specified, saves to {DEFAULT_PROCESS_DIR}/ directory with '_cropped' suffix.",
)
@click.option(
    "-s",
    "--trim-start",
    type=int,
    default=0,
    help=(
        "Start frame index for cropping (0-based, inclusive). "
        "Default 0 means no trimming from start."
    ),
)
@click.option(
    "-e",
    "--trim-end",
    type=int,
    default=0,
    help=(
        "End frame index for cropping (0-based, inclusive). "
        "Default 0 means no trimming from end."
    ),
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
    # Convert 0 values to None (meaning "no trimming")
    # This allows default=0 to mean "don't trim" while still allowing
    # explicit frame index 0 if needed (though unlikely)
    trim_start_val = None if trim_start == 0 else trim_start
    trim_end_val = None if trim_end == 0 else trim_end

    # If both are None (or both were 0), no trimming will occur
    if trim_start_val is None and trim_end_val is None:
        click.echo("No trimming specified (both start and end are 0). Copying entire video.")

    # Validate video/metadata match at the beginning
    is_valid, error_msg, csv_df = validate_video_metadata_match(input)

    if not is_valid:
        if error_msg and "not found" in error_msg.lower():
            if click.confirm(
                f"{error_msg}. Do you want to continue without generating "
                "output CSV metadata?",
                default=False,
            ):
                logger.warning(
                    f"{error_msg}. "
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
                logger.warning(f"{error_msg}. Proceeding anyway.")
            else:
                raise click.ClickException(f"{error_msg}. Cannot proceed.")

    # Resolve output path - use organized mio_process directory if not specified
    input_path = Path(input)
    output_arg = output if output is not None else DEFAULT_PROCESS_DIR
    output_path = resolve_output_path(input_path, "_cropped", output_arg)
    
    output_path_obj = crop_run(
        input,
        output_path=str(output_path),
        csv_validation_result=(is_valid, csv_df),
        trim_start=trim_start_val,
        trim_end=trim_end_val,
    )

    # Validate frame count alignment after cropping
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
    type=click.Path(),
    default=None,
    help="Path to the output stitched video file or directory. "
    f"If not specified, uses {DEFAULT_PROCESS_DIR}/ directory with '_stitched' suffix.",
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
    output: Optional[str],
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

    # Resolve output path - use organized mio_process directory if not specified
    # Use the first input's stem as base name for stitching
    first_input = Path(inputs[0])
    output_arg = output if output is not None else DEFAULT_PROCESS_DIR
    output_path_obj = resolve_output_path(first_input, "_stitched", output_arg)
    
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
    type=click.Path(),
    default=None,
    help="Base path for output files (stitched, cropped, denoised) or directory. "
    f"If not specified, uses {DEFAULT_PROCESS_DIR}/ directory.",
)
@click.option(
    "-c",
    "--denoise_config",
    required=True,
    type=str,
    help="Path to the YAML processing configuration file.",
)
@click.option(
    "-s",
    "--trim-start",
    type=int,
    default=0,
    help="Number of frames to remove from the start (default: 0).",
)
@click.option(
    "-e",
    "--trim-end",
    type=int,
    default=0,
    help="Number of frames to remove from the end (default: 0).",
)
@click.option(
    "--fps",
    type=int,
    default=20,
    help="Frames per second for stitched video.",
)
def workflow(
    inputs: tuple,
    output: Optional[str],
    denoise_config: str,
    trim_start: int,
    trim_end: int,
    fps: int,
) -> None:
    """
    Complete workflow: stitch → trim → denoise with validation at each step.
    """
    # Resolve output path - use organized mio_process directory if not specified
    first_input = Path(inputs[0])
    if output is None:
        # Use organized directory structure
        output_dir = Path.cwd() / DEFAULT_PROCESS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = first_input.stem
    else:
        output_path = Path(output).expanduser()
        if output_path.is_dir() or not output_path.suffix:
            # It's a directory, generate base name
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            output_dir = output_path
            output_stem = first_input.stem
        else:
            # It's a file path, use as base
            output_dir = output_path.parent
            output_stem = output_path.stem

    import cv2

    # Step 1: Stitch
    stitched_dir = output_dir / "stitched"
    stitched_dir.mkdir(parents=True, exist_ok=True)
    stitched_video = stitched_dir / f"{output_stem}_stitched.avi"
    logger.info("Stitching videos...")

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

    # Create debug output paths in stitched/debug/ directory
    debug_dir = stitched_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_video_path = debug_dir / f"{output_stem}_debug.avi"
    debug_csv_path = debug_dir / f"{output_stem}_debug.csv"

    # Create video writers
    combined_video_writer = VideoWriter(path=output_path_obj, fps=fps)
    debug_video_writer = VideoWriter(path=debug_video_path, fps=fps)

    # Create bundle and stitch
    recording_bundle = RecordingDataBundle(
        recordings=recordings,
        combined_video_writer=combined_video_writer,
        debug_video_writer=debug_video_writer,
        combined_csv_path=output_csv_path,
        debug_csv_path=debug_csv_path,
    )

    logger.info(f"Stitching {len(recordings)} recordings...")
    recording_bundle.stitch_recordings()

    # Validate frame count alignment after stitching
    is_aligned, alignment_error = validate_frame_count_alignment(output_path_obj)
    if not is_aligned:
        raise click.ClickException(
            f"Frame count alignment failed after stitching: {alignment_error}"
        )
    logger.info(f"✓ Frame count alignment verified: {output_path_obj}")
    logger.info(f"✓ Saved stitched video: {output_path_obj}")
    logger.info(f"✓ Saved stitched metadata: {output_csv_path}")

    if trim_start == 0 and trim_end == 0:
        logger.info("Skipping trim (both start and end are 0)...")
        actual_cropped_video = stitched_video
        logger.info("✓ No trimming needed, using stitched video as-is")
    else:
        cropped_dir = output_dir / "cropped"
        cropped_dir.mkdir(parents=True, exist_ok=True)
        cropped_video = cropped_dir / f"{output_stem}_cropped.avi"
        logger.info("Trimming video...")

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

        logger.info(
            f"Trimming: removing first {trim_start} frames and last {trim_end} frames "
            f"(keeping frames {crop_start}-{crop_end})"
        )

        # Validate input before cropping
        is_valid, error_msg, csv_df = validate_video_metadata_match(str(stitched_video))
        if not is_valid:
            raise click.ClickException(f"Cannot crop: {error_msg}")

        actual_cropped_video = crop_run(
            str(stitched_video),
            output_path=str(cropped_video),
            csv_validation_result=(is_valid, csv_df),
            trim_start=crop_start,
            trim_end=crop_end,
        )

        # Validate frame count alignment after cropping
        is_aligned, alignment_error = validate_frame_count_alignment(actual_cropped_video)
        if not is_aligned:
            raise click.ClickException(
                f"Frame count alignment failed after cropping: {alignment_error}"
            )
        logger.info(f"✓ Frame count alignment verified: {actual_cropped_video}")
        logger.info(f"✓ Saved cropped video: {actual_cropped_video}")
        logger.info(f"✓ Saved cropped metadata: {actual_cropped_video.with_suffix('.csv')}")

    # Step 3: Denoise
    logger.info("Denoising video...")

    # Validate input before denoising
    is_valid, error_msg, csv_df = validate_video_metadata_match(str(actual_cropped_video))
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
    
    # Override output_dir to use organized structure in workflow
    # Main denoised outputs go to denoised/ directory
    denoised_dir = output_dir / "denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    denoise_config_parsed.output_dir = str(denoised_dir)
    
    # Intermediate/debug files go to denoised/debug/ directory
    debug_dir = denoised_dir / "debug"
    
    denoise_run(
        str(actual_cropped_video),
        denoise_config_parsed,
        csv_validation_result=(is_valid, csv_df),
        debug_dir=debug_dir,
    )

    # Final validation and output summary
    # The denoise output is in the config output_dir, need to find it
    output_dir_denoise = Path(denoise_config_parsed.output_dir)
    if not output_dir_denoise.is_absolute():
        output_dir_denoise = Path.cwd() / output_dir_denoise
    
    # Find the main output video - the patch output is the primary one with CSV
    # Try different naming patterns based on what processors were enabled
    cropped_stem = Path(actual_cropped_video).stem
    output_videos = list(output_dir_denoise.glob(f"{cropped_stem}_patch.avi"))
    if not output_videos:
        output_videos = list(output_dir_denoise.glob(f"{cropped_stem}_output.avi"))
    if not output_videos:
        output_videos = list(output_dir_denoise.glob(f"{cropped_stem}_freq_mask.avi"))
    if not output_videos:
        # Fallback: try to find any .avi file in the denoised directory
        output_videos = list(output_dir_denoise.glob("*.avi"))
    
    if output_videos:
        # Use the first matching video (prefer _patch, then _output, then _freq_mask, then any)
        final_video = output_videos[0]
        # Find corresponding CSV - try multiple naming patterns
        # First try the expected pattern based on video name
        final_csv = final_video.with_suffix(".csv")
        if not final_csv.exists():
            # Try with _patch suffix if video doesn't have it
            if "_patch" not in final_video.stem:
                final_csv = output_dir_denoise / f"{cropped_stem}_patch.csv"
            else:
                final_csv = output_dir_denoise / f"{cropped_stem}_patch.csv"
        if not final_csv.exists():
            # Try with _metadata suffix
            final_csv = output_dir_denoise / f"{final_video.stem}_metadata.csv"
        if not final_csv.exists():
            final_csv = final_video.with_suffix(".csv")
        if not final_csv.exists():
            # Try with _metadata suffix
            final_csv = final_video.parent / f"{final_video.stem}_metadata.csv"
        
        logger.info("Final validation...")
        
        # Only validate if CSV exists
        if final_csv.exists():
            is_aligned, alignment_error = validate_frame_count_alignment(final_video)
            if not is_aligned:
                raise click.ClickException(
                    f"Frame count alignment failed after denoising: {alignment_error}"
                )
            logger.info(
                f"✓ Final frame count alignment verified: {final_video}"
            )
        else:
            logger.warning(
                f"CSV file not found for {final_video}, "
                "skipping alignment validation"
            )
        
        logger.info(f"✓ Saved denoised video: {final_video}")
        if final_csv.exists():
            logger.info(f"✓ Saved denoised metadata: {final_csv}")
    else:
        logger.warning("Could not find denoised output video for validation.")

    logger.info("Workflow completed")
    logger.info(f"Stitched: {stitched_video}")
    if trim_start != 0 or trim_end != 0:
        logger.info(f"Cropped: {actual_cropped_video}")
    else:
        logger.info("Cropped: (skipped)")
    if output_videos:
        logger.info(f"Denoised: {final_video}")

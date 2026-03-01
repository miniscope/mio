"""
Command line interface for offline video pre-processing.
"""

import shutil
from pathlib import Path
from typing import List, Optional

import click

from mio.exceptions import VideoMetadataError
from mio.io import VideoReader, VideoWriter
from mio.logging import init_logger
from mio.models.process import DenoiseConfig
from mio.process.stitch import RecordingData, RecordingDataBundle
from mio.process.video import crop_run, denoise_run
from mio.utils import (
    DEFAULT_PROCESS_DIR,
    extract_mismatch_details,
    resolve_output_path,
    validate_video_metadata_match,
)

logger = init_logger("mio.cli.process")


def _validate_with_prompt(video_path: str) -> Optional[object]:
    """Validate video-metadata match, prompting user on failure.

    Returns the validated DataFrame, or None if the user chose to continue
    without CSV metadata.
    """
    try:
        return validate_video_metadata_match(video_path)
    except VideoMetadataError as e:
        if e.csv_df is None:
            if click.confirm(
                f"{e}. Do you want to continue without generating output CSV metadata?",
                default=False,
            ):
                logger.warning(f"{e}. Continuing without CSV metadata generation.")
                return None
            raise click.ClickException(f"{e}. Cannot proceed without CSV.") from None
        else:
            if click.confirm(
                f"{e}. This may indicate a mismatch between the video and CSV. "
                "Do you want to continue anyway?",
                default=False,
            ):
                logger.warning(f"{e}. Proceeding anyway.")
                return e.csv_df
            raise click.ClickException(f"{e}. Cannot proceed.") from None


def _build_recordings(inputs: tuple) -> List[RecordingData]:
    """Build RecordingData list from input video paths, checking for companion CSVs."""
    recordings: List[RecordingData] = []
    for video_path in inputs:
        video_path_obj = Path(video_path)
        csv_path_obj = video_path_obj.with_suffix(".csv")
        if not csv_path_obj.exists():
            raise click.ClickException(f"CSV file not found for {video_path}: {csv_path_obj}")
        recordings.append(RecordingData(video_path=video_path_obj, csv_path=csv_path_obj))
    return recordings


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
    csv_df = _validate_with_prompt(input)

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)

    if output_dir is not None:
        denoise_config_parsed.output_dir = str(Path(output_dir).expanduser())
    elif denoise_config_parsed.output_dir is None:
        default_output_dir = Path.cwd() / DEFAULT_PROCESS_DIR
        default_output_dir.mkdir(parents=True, exist_ok=True)
        denoise_config_parsed.output_dir = str(default_output_dir)

    denoise_run(
        input,
        denoise_config_parsed,
        csv_df=csv_df,
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
    trim_start: int,
    trim_end: int,
) -> None:
    """
    Crop a video file by trimming frames.
    """
    # 0 means "don't trim"
    trim_start_val = None if trim_start == 0 else trim_start
    trim_end_val = None if trim_end == 0 else trim_end

    if trim_start_val is None and trim_end_val is None:
        click.echo("No trimming specified (both start and end are 0). Copying entire video.")

    csv_df = _validate_with_prompt(input)

    input_path = Path(input)
    output_arg = output if output is not None else DEFAULT_PROCESS_DIR
    output_path = resolve_output_path(input_path, "_cropped", output_arg)

    cropped_output = crop_run(
        input,
        output_path=str(output_path),
        csv_df=csv_df,
        trim_start=trim_start_val,
        trim_end=trim_end_val,
    )

    try:
        validate_video_metadata_match(cropped_output)
    except VideoMetadataError as e:
        raise click.ClickException(f"Frame count alignment failed after cropping: {e}") from None
    click.echo(f"✅ Frame count alignment verified: {cropped_output}")


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

    recordings = _build_recordings(inputs)

    first_input_path = Path(inputs[0])
    output_arg = output if output is not None else DEFAULT_PROCESS_DIR
    stitched_video_path = resolve_output_path(first_input_path, "_stitched", output_arg)

    output_csv_path = stitched_video_path.with_suffix(".csv")

    debug_video_path = Path(debug_video) if debug_video else None
    debug_csv_path = Path(debug_csv) if debug_csv else None

    stitched_video_writer = VideoWriter(path=stitched_video_path, fps=fps)
    debug_video_writer = VideoWriter(path=debug_video_path, fps=fps) if debug_video_path else None

    recording_bundle = RecordingDataBundle(
        recordings=recordings,
        stitched_video_writer=stitched_video_writer,
        debug_video_writer=debug_video_writer,
        combined_csv_path=output_csv_path,
        debug_csv_path=debug_csv_path,
    )

    click.echo(f"Stitching {len(recordings)} recordings...")
    recording_bundle.stitch_recordings()

    try:
        validate_video_metadata_match(stitched_video_path)
    except VideoMetadataError as e:
        raise click.ClickException(f"Frame count alignment failed after stitching: {e}") from None
    click.echo(f"✅ Frame count alignment verified: {stitched_video_path}")


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
    first_input_path = Path(inputs[0])
    if output is None:
        output_dir = Path.cwd() / DEFAULT_PROCESS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_stem = first_input_path.stem
    else:
        output_path = Path(output).expanduser()
        if output_path.is_dir() or not output_path.suffix:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            output_dir = output_path
            output_stem = first_input_path.stem
        else:
            output_dir = output_path.parent
            output_stem = output_path.stem

    click.echo("Validating input videos and CSV metadata...")
    validation_failures = []

    for video_path in inputs:
        video_path_obj = Path(video_path)
        try:
            validate_video_metadata_match(str(video_path_obj))
        except VideoMetadataError as e:
            validation_failures.append((video_path_obj, e))

    if validation_failures:
        click.echo("\n⚠️  Frame number validation found mismatches:")
        click.echo("=" * 70)

        for video_path_obj, error in validation_failures:
            click.echo(f"\n📹 Video: {video_path_obj.name}")

            details = extract_mismatch_details(video_path_obj, error.csv_df)
            if "missing_count" in details:
                click.echo("   ⚠️  Frame mismatch detected:")
                click.echo(f"      • Video has {details['video_frame_count']} frames")
                click.echo(f"      • CSV has {details['csv_frame_count']} unique frame indices")
                click.echo(f"      • Missing {details['missing_count']} frame(s) in CSV")

                if details.get("missing_ranges"):
                    ranges_str = ", ".join(details["missing_ranges"][:10])
                    if len(details["missing_ranges"]) > 10:
                        ranges_str += f" ... ({len(details['missing_ranges'])} total ranges)"
                    click.echo(f"      • Missing frame ranges: {ranges_str}")
            else:
                click.echo(f"   ❌ {error}")

        click.echo("\n" + "=" * 70)
        click.echo(
            "\n⚠️  Warning: Missing frames in CSV metadata will be skipped during processing.\n"
            "   This may result in incomplete output or data loss.\n"
        )

        if not click.confirm(
            "Do you want to continue with the workflow despite these mismatches?",
            default=False,
        ):
            raise click.ClickException(
                "Workflow cancelled due to frame number validation failures."
            )

        logger.warning(
            f"Proceeding with workflow despite {len(validation_failures)} "
            "validation failure(s). Missing frames will be skipped."
        )
    else:
        click.echo("✅ [input] All input videos validated successfully")

    stitched_dir = output_dir / "stitched"
    stitched_dir.mkdir(parents=True, exist_ok=True)
    stitched_video_path = stitched_dir / f"{output_stem}_stitched.avi"

    if len(inputs) == 1:
        click.echo("Only one input video provided, skipping stitching...")
        input_video_path = Path(inputs[0])
        input_csv_path = input_video_path.with_suffix(".csv")

        click.echo(f"Copying single video to stitched directory: {stitched_video_path}")
        shutil.copy2(input_video_path, stitched_video_path)
        shutil.copy2(input_csv_path, stitched_video_path.with_suffix(".csv"))

        click.echo(
            f"✅ [stitch] Using single input video as stitched output: {stitched_video_path}"
        )
    else:
        recordings = _build_recordings(inputs)

        output_csv_path = stitched_video_path.with_suffix(".csv")

        debug_dir = stitched_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_video_path = debug_dir / f"{output_stem}_debug.avi"
        debug_csv_path = debug_dir / f"{output_stem}_debug.csv"

        stitched_video_writer = VideoWriter(path=stitched_video_path, fps=fps)
        debug_video_writer = VideoWriter(path=debug_video_path, fps=fps)

        recording_bundle = RecordingDataBundle(
            recordings=recordings,
            stitched_video_writer=stitched_video_writer,
            debug_video_writer=debug_video_writer,
            combined_csv_path=output_csv_path,
            debug_csv_path=debug_csv_path,
        )

        click.echo(f"Stitching {len(recordings)} recordings...")
        recording_bundle.stitch_recordings()

        try:
            validate_video_metadata_match(stitched_video_path)
        except VideoMetadataError as e:
            raise click.ClickException(
                f"Frame count alignment failed after stitching: {e}"
            ) from None
        click.echo(f"✅ [stitch] Frame count alignment verified: {stitched_video_path}")

    if trim_start == 0 and trim_end == 0:
        click.echo("Skipping trim (both start and end are 0)...")
        actual_cropped_video = stitched_video_path
        click.echo("✅ [crop] No trimming needed, using stitched video as-is")
    else:
        cropped_dir = output_dir / "cropped"
        cropped_dir.mkdir(parents=True, exist_ok=True)
        cropped_video = cropped_dir / f"{output_stem}_cropped.avi"
        click.echo("Trimming video...")

        reader = VideoReader(str(stitched_video_path))
        total_frames = reader.frame_count
        reader.release()

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

        try:
            csv_df = validate_video_metadata_match(str(stitched_video_path))
        except VideoMetadataError as e:
            raise click.ClickException(f"Cannot crop: {e}") from None

        actual_cropped_video = crop_run(
            str(stitched_video_path),
            output_path=str(cropped_video),
            csv_df=csv_df,
            trim_start=crop_start,
            trim_end=crop_end,
        )

        try:
            validate_video_metadata_match(actual_cropped_video)
        except VideoMetadataError as e:
            raise click.ClickException(
                f"Frame count alignment failed after cropping: {e}"
            ) from None
        click.echo(f"✅ [crop] Frame count alignment verified: {actual_cropped_video}")

    click.echo("Denoising video...")

    csv_df = None
    try:
        csv_df = validate_video_metadata_match(str(actual_cropped_video))
    except VideoMetadataError as e:
        if e.csv_df is None:
            raise click.ClickException(f"{e}. Cannot proceed without CSV.") from None
        if not click.confirm(
            f"{e}. This may indicate a mismatch between the video and CSV. "
            "Do you want to continue anyway?",
            default=False,
        ):
            raise click.ClickException(f"{e}. Cannot proceed.") from None
        csv_df = e.csv_df

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)

    denoised_dir = output_dir / "denoised"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    denoise_config_parsed.output_dir = str(denoised_dir)

    debug_dir = denoised_dir / "debug"

    denoise_run(
        str(actual_cropped_video),
        denoise_config_parsed,
        csv_df=csv_df,
        debug_dir=debug_dir,
    )

    output_dir_denoise = Path(denoise_config_parsed.output_dir)
    if not output_dir_denoise.is_absolute():
        output_dir_denoise = Path.cwd() / output_dir_denoise

    cropped_stem = Path(actual_cropped_video).stem
    output_videos = list(output_dir_denoise.glob(f"{cropped_stem}_patch.avi"))
    if not output_videos:
        output_videos = list(output_dir_denoise.glob(f"{cropped_stem}_output.avi"))
    if not output_videos:
        output_videos = list(output_dir_denoise.glob(f"{cropped_stem}_freq_mask.avi"))
    if not output_videos:
        output_videos = list(output_dir_denoise.glob("*.avi"))

    if output_videos:
        final_video = output_videos[0]
        final_csv = final_video.with_suffix(".csv")
        if not final_csv.exists():
            final_csv = output_dir_denoise / f"{cropped_stem}_patch.csv"
        if not final_csv.exists():
            final_csv = output_dir_denoise / f"{final_video.stem}_metadata.csv"
        if not final_csv.exists():
            final_csv = final_video.parent / f"{final_video.stem}_metadata.csv"

        click.echo("Final validation...")

        if final_csv.exists():
            try:
                validate_video_metadata_match(final_video)
            except VideoMetadataError as e:
                raise click.ClickException(
                    f"Frame count alignment failed after denoising: {e}"
                ) from None
            click.echo(f"✅ [denoise] Frame count alignment verified: {final_video}")
        else:
            logger.warning(
                f"CSV file not found for {final_video}, " "skipping alignment validation"
            )
    else:
        logger.warning("Could not find denoised output video for validation.")

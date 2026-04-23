"""
Command line interface for offline video pre-processing.
"""

import re
from pathlib import Path

import click

from mio.logging import init_logger
from mio.models.dataset import Recording
from mio.models.process import DenoiseConfig
from mio.process.stitch import concat_recordings
from mio.process.stitch import stitch as run_stitch
from mio.process.video import denoise as run_denoise
from mio.process.video import remove_frames as run_remove_frames
from mio.process.video import trim as run_trim

logger = init_logger("mio.cli.process")


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
        "Output directory for denoised files. " "If not specified, uses directory of input video."
    ),
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite any existing files",
)
def denoise(input: Path, denoise_config: str, output_dir: str | None, force: bool = False) -> None:
    """
    Denoise a video file by detecting and removing frames with noisy areas.

    Processing steps (noisy area detection, frequency masking, minimum
    projection subtraction) are configured via the YAML config file.
    """
    input = Path(input)
    recording = Recording.from_video(input)

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)

    if output_dir is not None:
        denoise_config_parsed.output_dir = str(Path(output_dir).expanduser())
    elif denoise_config_parsed.output_dir is None:
        default_output_dir = input.parent
        denoise_config_parsed.output_dir = str(default_output_dir)

    run_denoise(input, denoise_config_parsed, csv_df=recording.metadata, progress=True, force=force)


@process.command()
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the video file to trim.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Path to the output video file or directory. If a directory, "
    "the output filename will be generated from the input filename. "
    "If not specified, saves to input file name with '_trimmed' suffix.",
)
@click.option(
    "-s",
    "--trim-start",
    type=int,
    default=0,
    help="Number of frames to remove from the beginning. Default: 0.",
)
@click.option(
    "-e",
    "--trim-end",
    type=int,
    default=0,
    help="Number of frames to remove from the end. Default: 0.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite any existing files",
)
def trim(
    input: str, output: Path | None, trim_start: int, trim_end: int, force: bool = False
) -> None:
    """
    Crop a video by removing frames from the start and/or end.

    Also trims and renumbers the companion CSV metadata to match.
    """
    input_path = Path(input)
    recording = Recording.from_video(input_path)
    if not output:
        output_path = input_path.parent / (input_path.stem + "_trimmed" + input_path.suffix)
    elif (output := Path(output)).is_dir():
        output_path = output / (input_path.stem + "_trimmed" + input_path.suffix)
    else:
        output_path = Path(output)

    trimmed_output = run_trim(
        input_path,
        output_path=output_path,
        csv_df=recording.metadata,
        start=trim_start,
        end=trim_end,
        progress=True,
        force=force,
    )
    click.echo(f"Cropped output written to {trimmed_output}")


@process.command(name="remove-frames")
@click.option(
    "-i",
    "--input",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the video file. Each requires a .csv with the same stem name.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to the output video file."
    "If not specified, add a '_removed' suffix and write to same directory",
)
@click.option(
    "-f",
    "--frames",
    required=True,
    type=str,
    help="Comma-separated list of 0-based frame indices to remove (e.g. '0,5,10,42').",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
)
def remove_frames(input: str, output: str | None, frames: str, force: bool = False) -> None:
    """
    Remove specific frames by index from a video.

    A manual cleanup step for removing individual bad frames after reviewing
    the output. Also updates the companion CSV metadata to match.
    """
    input = Path(input)
    recording = Recording.from_video(input)
    if recording.metadata is None:
        click.echo("Recording has no matching metadata! Just removing frames from video")

    output = Path(output) if output else input.with_stem(input.stem + "_removed")

    frame_indices = [int(x.strip()) for x in frames.split(",")]

    run_remove_frames(
        recording,
        remove_indices=frame_indices,
        output_path=output,
        progress=True,
        force=force,
    )
    click.echo(f"Video written to {output} with frames {frames} removed from {input}")


@process.command()
@click.option(
    "-d",
    "--directory",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing .avi segment files. All .avi files with companion "
    ".csv files will be discovered and sorted in natural numeric order.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to the output concatenated video file or directory. "
    "If not specified, saves next to video with '_combined' suffix.",
)
@click.option(
    "--fps",
    type=int,
    default=20,
    help="Frames per second for output video.",
)
def concat(
    directory: str,
    output: str | None,
    fps: int,
) -> None:
    """
    Concatenate sequential recording segments from one DAQ into a single video.

    Discovers all .avi files in the given directory (that have companion .csv files),
    sorts them by filename, and concatenates them into a single video + CSV with
    contiguous reconstructed_frame_index.

    Use this to combine multiple segment files (e.g. long-2.avi, long-3.avi, ...)
    from the same DAQ before stitching across DAQs.
    """
    dir_path = Path(directory)

    def _natural_sort_key(path: Path) -> list:
        """Sort by splitting name into text and numeric parts for natural ordering."""
        return [
            int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)
        ]

    avi_files = sorted(dir_path.glob("*.avi"), key=_natural_sort_key)

    def _find_companion_csv(avi_path: Path) -> Optional[Path]:
        """Find companion CSV, handling name mismatches like long-8-002.avi -> long-8.csv."""
        # Try exact stem match first
        csv_path = avi_path.with_suffix(".csv")
        if csv_path.exists():
            return csv_path
        # Try stripping trailing numeric suffixes: long-8-002.avi -> long-8.csv
        stem = avi_path.stem
        stripped = re.sub(r"-\d+$", "", stem)
        while stripped != stem:
            csv_path = avi_path.parent / f"{stripped}.csv"
            if csv_path.exists():
                return csv_path
            stem = stripped
            stripped = re.sub(r"-\d+$", "", stem)
        return None

    # Filter to only AVIs that have a companion CSV
    valid_pairs: list[tuple[Path, Path]] = []
    for avi in avi_files:
        csv_path = _find_companion_csv(avi)
        if csv_path is not None:
            valid_pairs.append((avi, csv_path))
        else:
            click.echo(f"  Skipping {avi.name} (no companion .csv found)")

    if len(valid_pairs) < 2:
        raise click.ClickException(
            f"Need at least 2 .avi files with companion .csv files in {directory}, "
            f"found {len(valid_pairs)}."
        )

    click.echo(f"Found {len(valid_pairs)} segments in {directory}:")
    for avi, csv in valid_pairs:
        click.echo(f"  {avi.name} -> {csv.name}")

    recordings = [RecordingData(video_path=avi, csv_path=csv) for avi, csv in valid_pairs]

    first_input_path = valid_pairs[0][0]
    output_arg = output if output is not None else DEFAULT_PROCESS_DIR
    combined_video_path = resolve_output_path(first_input_path, "_combined", output_arg)
    combined_csv_path = combined_video_path.with_suffix(".csv")

    click.echo(f"Concatenating {len(recordings)} segments...")
    concat_recordings(
        recordings=recordings,
        output_video_path=combined_video_path,
        output_csv_path=combined_csv_path,
        fps=fps,
    )

    try:
        validate_video_metadata_match(combined_video_path)
    except VideoMetadataError as e:
        raise click.ClickException(
            f"Frame count alignment failed after concatenation: {e}"
        ) from None
    click.echo(f"✅ Frame count alignment verified: {combined_video_path}")


@process.command()
@click.option(
    "-i",
    "--inputs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Paths to video files. Each requires a .csv with the same stem name.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for output videos and metadata. If none provided, same as the inputs.",
)
@click.option(
    "--debug-video",
    default=False,
    is_flag=True,
    help="Output path for debug video showing frame comparisons.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite any existing files",
)
def stitch(
    inputs: tuple, output: Path | None = None, debug_video: bool = False, force: bool = False
) -> None:
    """
    Stitch multiple video recordings into one by selecting the best frame
    for each device timestamp using metadata scoring and edge detection.

    Currently tested with 2 inputs. More may work but are untested.
    """
    if len(inputs) < 2:
        raise click.ClickException("At least 2 input videos are required for stitching.")

    recordings = [Recording.from_video(Path(p)) for p in inputs]
    stitched = run_stitch(
        recordings, debug_video=debug_video, output_dir=output, progress=True, force=force
    )
    click.echo(f"Stitched videos to {stitched.video.path}")


@process.command()
@click.option(
    "-i",
    "--inputs",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Paths to video files. Each requires a .csv with the same stem name.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    default=None,
    help="Base path for output files (stitched, cropped, denoised) or directory. "
    "If not specified, uses input directory.",
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
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite any existing files",
)
def workflow(
    inputs: tuple,
    output: str | None,
    denoise_config: str,
    trim_start: int,
    trim_end: int,
    force: bool = False,
) -> None:
    """
    Complete workflow: stitch → trim → denoise with validation at each step.
    """
    inputs = [Path(i) for i in inputs]
    if output is None:
        output_dir = inputs[0].parent
    else:
        output_dir = Path(output).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

    if len(inputs) == 1:
        click.echo("Only one input video provided, skipping stitching")
        stitched_video = inputs[0]
    else:
        click.echo("Stitching videos...")
        recordings = [Recording.from_video(p) for p in inputs]
        stitched = run_stitch(recordings, output_dir=output_dir, progress=True, force=force)
        stitched_video = stitched.video.path

    if trim_start == 0 and trim_end == 0:
        click.echo("Not trimming, trim start and end both zero")
        trimmed_video = stitched_video
    else:
        click.echo("Trimming video...")
        trimmed_video = run_trim(
            stitched_video, output_dir, start=trim_start, end=trim_end, progress=True, force=force
        )

    trimmed = Recording.from_video(trimmed_video)
    if trimmed.metadata is None:
        raise FileNotFoundError(f"No metadata csv found for video {trimmed_video}")

    denoise_config_parsed = DenoiseConfig.from_any(denoise_config)
    final_video = run_denoise(
        trimmed_video,
        denoise_config_parsed,
        csv_df=trimmed.metadata,
        debug_dir=output_dir / "debug",
        progress=True,
        force=force,
    )
    click.echo(f"Processed video written to {final_video}")

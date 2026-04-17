"""
Command line interface for offline video pre-processing.
"""

from pathlib import Path

import click

from mio.logging import init_logger
from mio.models.dataset import Recording
from mio.models.process import DenoiseConfig
from mio.process.stitch import stitch as run_stitch
from mio.process.video import denoise as run_denoise
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
def denoise(
    input: Path,
    denoise_config: str,
    output_dir: str | None,
) -> None:
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

    run_denoise(input, denoise_config_parsed, csv_df=recording.metadata, progress=True)


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
def trim(
    input: str,
    output: Path | None,
    trim_start: int,
    trim_end: int,
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
    )
    click.echo(f"Cropped output written to {trimmed_output}")


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
def stitch(inputs: tuple, output: Path | None = None, debug_video: bool = False) -> None:
    """
    Stitch multiple video recordings into one by selecting the best frame
    for each device timestamp using metadata scoring and edge detection.

    Currently tested with 2 inputs. More may work but are untested.
    """
    if len(inputs) < 2:
        raise click.ClickException("At least 2 input videos are required for stitching.")

    recordings = [Recording.from_video(Path(p)) for p in inputs]
    stitched = run_stitch(recordings, debug_video=debug_video, output_dir=output, progress=True)
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
def workflow(
    inputs: tuple,
    output: str | None,
    denoise_config: str,
    trim_start: int,
    trim_end: int,
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
        stitched = run_stitch(recordings, output_dir=output_dir, progress=True)
        stitched_video = stitched.video.path

    if trim_start == 0 and trim_end == 0:
        click.echo("Not trimming, trim start and end both zero")
        trimmed_video = stitched_video
    else:
        click.echo("Trimming video...")
        trimmed_video = run_trim(
            stitched_video,
            output_dir,
            start=trim_start,
            end=trim_end,
            progress=True,
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
    )
    click.echo(f"Processed video written to {final_video}")

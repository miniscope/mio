"""
Command line interface for offline video pre-processing.
"""

from typing import Optional

import click

from mio.models.process import DenoiseConfig
from mio.process.video import denoise_run
from mio.utils import validate_video_metadata_match


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
    "-s",
    "--trim-start",
    type=int,
    default=None,
    help="Start frame index for trimming (0-based, inclusive).",
)
@click.option(
    "-e",
    "--trim-end",
    type=int,
    default=None,
    help="End frame index for trimming (0-based, inclusive).",
)
def denoise(
    input: str,
    denoise_config: str,
    trim_start: Optional[int],
    trim_end: Optional[int],
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
        trim_start=trim_start,
        trim_end=trim_end,
    )

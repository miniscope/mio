"""
CLI commands for recording video from USB camera.
"""

import os
from pathlib import Path
from typing import Optional

import click

from mio.behavior_cam import BehaviorCam
from mio.cli.common import ConfigIDOrPath
from mio.devices.usbcam import format_camera_info
from mio.devices.usbcam import list_cameras as list_available_cameras
from mio.models.usbcam import USBCameraRecordingConfig
from mio.ntp import prompt_ntp_sync


@click.group(invoke_without_command=True)
@click.option(
    "--list",
    "list_cameras",
    is_flag=True,
    help="List available cameras and exit",
)
@click.pass_context
def usbcam(ctx: click.Context, list_cameras: bool) -> None:
    """
    Command group for USB Camera
    """
    if list_cameras:
        cameras = list_available_cameras()
        if not cameras:
            click.echo("No cameras found")
        else:
            click.echo("Available cameras:")
            for idx, info in cameras.items():
                click.echo(f"  {format_camera_info(idx, info)}")
        ctx.exit()

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@usbcam.command()
@click.option(
    "-c",
    "--config",
    required=True,
    type=ConfigIDOrPath(),
    help=(
        "Either a config `id` or a path to USB camera config YAML file. "
        "If path is relative, treated as relative to the current directory, "
        "and then if no matching file is found, relative to the user `config_dir` "
        "(see `mio config --help`)."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    help="Override output directory from config (optional)",
)
@click.option(
    "-i",
    "--index",
    type=int,
    help="Specify camera index (optional)",
)
@click.option(
    "-b",
    "--binary_export",
    is_flag=True,
    help="Save raw frames to a .npz file alongside the video",
)
@click.option("--no-display", is_flag=True, help="Don't show video preview in real time")
def record(
    config: str,
    output_dir: Optional[str],
    index: Optional[int],
    binary_export: bool,
    no_display: bool,
) -> None:
    """Record video with Unix timestamp filename"""
    recording_config = USBCameraRecordingConfig.from_any(config)

    # Check NTP sync if configured
    if recording_config.ntp_server is not None:
        prompt_ntp_sync(
            recording_config.ntp_server, max_offset_seconds=recording_config.ntp_max_offset_seconds
        )

    # Override output_dir if provided via CLI
    if output_dir is not None:
        recording_config.output_dir = output_dir

    if index is not None:
        camera_index = index
    else:
        # Get available cameras and prompt for selection
        cameras = list_available_cameras()
        if not cameras:
            raise click.ClickException("No cameras found. Please connect a camera and try again.")

        click.echo("Available cameras:")
        for idx, info in cameras.items():
            click.echo(f"  {format_camera_info(idx, info)}")

        selected_index = click.prompt(
            "Select camera index",
            type=click.Choice([str(idx) for idx in cameras], case_sensitive=False),
            default=str(min(cameras.keys())),
        )
        camera_index = int(selected_index)

    # Compute binary export path if requested
    if binary_export:
        import time as _time

        binary_output = Path(recording_config.output_dir) / f"{int(_time.time())}.npz"
    else:
        binary_output = None

    behavior_cam = BehaviorCam(recording_config=recording_config, camera_index=camera_index)
    try:
        behavior_cam.capture(show_video=not no_display, capture_binary=binary_output)
    except Exception as e:
        click.echo(f"Error recording video: {e}", err=True)
        raise click.ClickException(f"Error recording video: {e}") from e


@usbcam.command()
@click.option(
    "-c",
    "--config",
    required=True,
    type=ConfigIDOrPath(),
    help="Either a config `id` or a path to USB camera config YAML file.",
)
@click.option(
    "-s",
    "--source",
    required=True,
    help="Path to .npz file with recorded frames",
    type=click.Path(exists=True),
)
@click.option(
    "-b",
    "--binary_export",
    is_flag=True,
    help="Save raw frames to a .npz file alongside the video",
)
@click.option("--no-display", is_flag=True, help="Don't show video preview in real time")
@click.pass_context
def test(
    ctx: click.Context, config: str, source: str, binary_export: bool, no_display: bool
) -> None:
    """
    Run BehaviorCam in testing mode, using USBCamMock rather than the actual device.
    """
    os.environ["BEHAVIORCAM_MOCKRUN"] = "just_placeholder"
    os.environ["PYTEST_USBCAM_DATA_FILE"] = str(source)

    ctx.invoke(
        record,
        config=config,
        output_dir=None,
        index=0,
        binary_export=binary_export,
        no_display=no_display,
    )

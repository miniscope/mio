"""
CLI commands for recording video from USB camera.
"""

from typing import Optional

import click

from mio.behavior_cam import BehaviorCam
from mio.cli.common import ConfigIDOrPath
from mio.devices.usbcam import ELPUVCCamera
from mio.models.usbcam import USBCameraRecordingConfig


@click.group()
def usbcam() -> None:
    """
    Command group for USB Camera
    """
    pass


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
def record(config: str, output_dir: Optional[str]) -> None:
    """Record video with Unix timestamp filename"""
    recording_config = USBCameraRecordingConfig.from_any(config)

    # Override output_dir if provided via CLI
    if output_dir is not None:
        recording_config.output_dir = output_dir

    behavior_cam = BehaviorCam(recording_config=recording_config)
    try:
        behavior_cam.capture(output_dir=output_dir)
    except Exception as e:
        click.echo(f"Error recording video: {e}", err=True)
        raise click.ClickException(f"Error recording video: {e}") from e


@usbcam.command()
def list_cameras() -> None:
    """List available cameras"""
    cameras = ELPUVCCamera.list_cameras()
    if not cameras:
        click.echo("No cameras found")
        return

    click.echo("Available cameras:")
    for idx, info in cameras.items():
        click.echo(f"  Index {idx}: {info['resolution']} @ {info['fps']} fps")

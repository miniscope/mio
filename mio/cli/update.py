"""
CLI for updating device over IR or UART.
"""

import click
import yaml

from mio.device_update import device_update
from mio.models.devupdate import DeviceCommand


@click.command()
@click.option(
    "-p",
    "--port",
    required=False,
    help="Serial port to connect to. Needed if multiple FTDI devices are connected.",
)
@click.option(
    "-i",
    "--device_id",
    required=False,
    default=0,
    type=int,
    help="[EXPERIMENTAL FEATURE] ID of the device to update. 0 (default) will update all devices.",
)
@click.option(
    "-k",
    "--key",
    required=False,
    type=click.Choice(["LED", "GAIN", "ROI_X", "ROI_Y", "SUBSAMPLE","TL"]),
    help="key to update.",
)
@click.option(
    "-v",
    "--value",
    required=False,
    type=int,
    help="Value to set. Must be used with --key and cannot be used with --restart.",
)
@click.option(
    "-b",
    "--batch",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "[EXPERIMENTAL FEATURE] YAML file that works as a batch file to update."
        "Specify key and value pairs in the file."
    ),
)
def update(port: str, key: str, value: int, device_id: int, batch: str) -> None:
    """
    Update device configuration.
    """
    # Check mutual exclusivity
    if (key and value is None) or (value and not key):
        raise click.UsageError("Both --key and --value are required if one is specified.")

    if batch and (key or value):
        raise click.UsageError("Options --key/--value and --batch are mutually exclusive.")
    if key and value is not None:
        device_update(port=port, key=key, value=value, device_id=device_id)
    elif batch:
        with open(batch) as f:
            batch_file = yaml.safe_load(f)
        for key, value in batch_file:
            device_update(port=port, key=key, value=value, device_id=device_id)
    else:
        raise click.UsageError("Either --key with --value or --restart must be specified.")


@click.command()
@click.option(
    "-p",
    "--port",
    required=False,
    help="Serial port to connect to. Needed if multiple FTDI devices are connected.",
)
@click.option(
    "-i",
    "--device_id",
    required=False,
    default=0,
    type=int,
    help="[EXPERIMENTAL FEATURE] ID of the device to update. 0 (default) will update all devices.",
)
@click.option(
    "--reboot",
    is_flag=True,
    type=bool,
    help="Restart the device.",
)
def device(port: str, device_id: int, reboot: bool) -> None:
    """
    Send device commands (e.g., reboot)
    """

    # Check mutual exclusivity
    if reboot:
        device_update(
            port=port, key="DEVICE", value=DeviceCommand.REBOOT.value, device_id=device_id
        )
    else:
        raise click.UsageError("Only --reboot is currently implemented.")

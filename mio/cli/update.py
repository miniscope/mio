"""
CLI for updating device over IR or UART.
"""

import click

from mio.cli.common import ConfigIDOrPath
from mio.device_update import device_update
from mio.models.devupdate import DeviceCommand
from mio.models.update import UpdateBatch


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
    type=click.Choice(["LED", "GAIN", "ROI_X", "ROI_Y", "SUBSAMPLE"]),
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
    type=ConfigIDOrPath(),
    help=(
        "[EXPERIMENTAL FEATURE] Batch config (ID or YAML path). "
        "Provide a list of device entries."
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
        batch_cfg = UpdateBatch.from_any(batch)
        for entry_port, entry_dev_id, k, v in batch_cfg.iter_updates(port):
            device_update(port=entry_port, key=k, value=v, device_id=entry_dev_id)
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

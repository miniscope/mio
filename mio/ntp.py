"""
NTP utilities for time synchronization checks.
"""

from typing import Tuple

import click
import ntplib

from mio.logging import init_logger

logger = init_logger("mio.ntp")


def query_ntp_sync(ntp_server: str, timeout: float = 3.0) -> Tuple[bool, float]:
    """
    Query the NTP server for the system time offset.

    Args:
        ntp_server: NTP server hostname or IP address
        timeout: Timeout for NTP query in seconds (default: 3.0)

    Returns:
        Tuple of (success: bool, offset_seconds: float)
        Returns (False, 0.0) if NTP query fails
    """
    try:
        client = ntplib.NTPClient()
        response = client.request(ntp_server, version=3, timeout=timeout)

        offset = abs(response.offset)

        return (True, offset)

    except (ntplib.NTPException, OSError, Exception):
        return (False, 0.0)


def prompt_ntp_sync(ntp_server: str, max_offset_seconds: float) -> None:
    """
    Check NTP sync and prompt user to proceed if not synchronized.
    This is a reusable helper for CLI commands - just call this with the ntp_server from config.

    Args:
        ntp_server: NTP server address from config
        max_offset_seconds: Maximum allowed time offset in seconds

    Raises:
        click.Abort: If user chooses not to proceed when sync is insufficient
    """
    logger.info(f"Checking time sync with NTP server: {ntp_server}")
    success, offset = query_ntp_sync(ntp_server)

    if not success:
        logger.warning(f"Could not query NTP server {ntp_server}.")
        if not click.confirm("System time may not be synchronized. Proceed anyway?"):
            raise click.Abort()
        return

    is_synced = offset <= max_offset_seconds
    if is_synced:
        logger.info(f"Time is synchronized with NTP server {ntp_server} (offset: {offset:.3f}s)")
    else:
        logger.warning(f"Time offset is {offset:.3f}s (max allowed: {max_offset_seconds:.3f}s).")
        if not click.confirm("System time may not be synchronized. Proceed anyway?"):
            raise click.Abort()

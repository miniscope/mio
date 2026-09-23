"""
NTP utilities for time synchronization checks.
"""

import socket

import click

try:
    import ntplib
except ImportError:
    ntplib = None  # type: ignore[assignment]

from mio.logging import init_logger

logger = init_logger("mio.ntp")


def _resolve_hostname(hostname: str) -> str:
    """
    Resolve hostname to IP address.

    This is necessary because ntplib has issues with mDNS .local hostnames
    on macOS (where .local is reserved for Bonjour/mDNS), even though they resolve
    correctly via socket.gethostbyname. Windows handles .local domains differently
    and doesn't have this issue, but resolving to IP first should work on all platforms.

    Args:
        hostname: Hostname or IP address

    Returns:
        IP address as string, or original string if resolution fails
    """
    try:
        ip_address = socket.gethostbyname(hostname)
        logger.info(f"Resolved {hostname} to {ip_address}")
        return ip_address
    except (socket.gaierror, OSError):
        logger.warning(f"Could not resolve hostname {hostname} to IP address.")
        return hostname


def query_ntp_sync(ntp_server: str, timeout: float = 3.0) -> tuple[bool, float]:
    """
    Query the NTP server for the system time offset.

    Args:
        ntp_server: NTP server hostname or IP address
        timeout: Timeout for NTP query in seconds (default: 3.0)

    Returns:
        Tuple of (success: bool, offset_seconds: float)
        Returns (False, 0.0) if NTP query fails

    Raises:
        ImportError: If ntplib is not installed. Install with: pip install mio[ntp]
    """
    if ntplib is None:
        raise ImportError(
            "ntplib is required for NTP functionality. Install it with: pip install mio[ntp]"
        )
    try:
        # Resolve hostname to IP to work around ntplib issues with mDNS .local hostnames on macOS
        resolved_server = _resolve_hostname(ntp_server)
        client = ntplib.NTPClient()
        response = client.request(resolved_server, version=3, timeout=timeout)

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
        ImportError: If ntplib is not installed. Install with: pip install mio[ntp]
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
        logger.info(
            f"Time is synchronized with NTP server {ntp_server} (offset: {offset * 1000:.3f}ms)"
        )
    else:
        logger.warning(
            f"Time offset: {offset * 1000:.3f}ms, max allowed: {max_offset_seconds * 1000:.3f}ms."
        )
        if not click.confirm("System time may not be synchronized. Proceed anyway?"):
            raise click.Abort()

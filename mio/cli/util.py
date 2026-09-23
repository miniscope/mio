"""
Some random utility functions. This should be organized elsewhere but just throwing it here for now.
"""

from pathlib import Path

import click

from mio.utils import hash_video


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def hash(path: str) -> None:
    """
    Hash a file
    """

    if Path(path).suffix in [".avi", ".mp4"]:
        path = Path(path)
        click.echo(hash_video(path))
    else:
        click.echo("Only video files are supported now.")

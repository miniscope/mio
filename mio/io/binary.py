"""Raw bytes i/o"""

from pathlib import Path


def append_binary(path: Path | None, data: bytes) -> None:
    """Just append some binary to a path!"""
    if path is None:
        # FIXME: Still working how we want enabling/disabling binary i/o to look like in streamdev
        return
    with open(path, "ab") as f:
        f.write(data)

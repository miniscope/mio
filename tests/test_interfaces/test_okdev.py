from pathlib import Path

import pytest

from mio.interfaces.mocks import okDevMock


@pytest.fixture
def _tmp_buffer(tmp_path: Path, set_okdev_input) -> bytes:
    buffer = b"12345"
    buffer_rep = b"12345" * 3
    data_path = tmp_path / "data.bin"
    with open(data_path, "wb") as f:
        f.write(buffer_rep)
    set_okdev_input(data_path)
    return buffer


def test_okdev_iter(_tmp_buffer: bytes) -> None:
    """
    sanity check that the __iter__ method for iteration on the okdev
    does iter things
    """
    dev = okDevMock(read_length=5)
    got_buffers = 0
    for buf in dev:
        assert buf == _tmp_buffer
        got_buffers += 1
    assert got_buffers == 3


def test_okdev_next(_tmp_buffer: bytes) -> None:
    """
    the okDevMock.__next__ method (and by proxy okDev) does next things
    """
    dev = okDevMock(read_length=5)
    buffers = []
    buffers.append(next(dev))
    buffers.append(next(dev))
    buffers.append(next(dev))
    with pytest.raises(StopIteration):
        next(dev)

    assert all([b == _tmp_buffer for b in buffers])

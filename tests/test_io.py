import pytest
import tempfile
from pathlib import Path
import os
import csv

import numpy as np
import warnings

from mio.models.sdcard import SDBufferHeader
from mio.io.sdcard import SDCard
from mio.io.file import BufferedCSVWriter
from mio.exceptions import EndOfRecordingException
from mio.models.frames import SDCardFrame
from mio.utils import hash_file, hash_video

from .fixtures import wirefree, wirefree_battery


@pytest.fixture
def tmp_csvfile(tmp_path):
    """
    Creates a temporary file for testing.
    """
    return tmp_path / "test.csv"


def test_csvwriter_initialization(tmp_csvfile):
    """
    Test that the BufferedCSVWriter initializes correctly.
    """
    writer = BufferedCSVWriter(tmp_csvfile, ["hey", "sup"], buffer_size=10)
    assert writer.file_path == tmp_csvfile
    assert writer.buffer_size == 10
    assert writer.buffer == [["hey", "sup"]]


def test_csvwriter_append_and_flush(tmp_csvfile):
    """
    Test that the BufferedCSVWriter appends to the buffer and flushes it when full.
    """
    writer = BufferedCSVWriter(tmp_csvfile, [1, 2, 3], buffer_size=3)
    # header added to buffer on init
    assert len(writer.buffer) == 1
    writer.append({1:1, 2:2, 3:3})
    assert len(writer.buffer) == 2

    writer.append({1:4, 2:5, 3:6})
    assert len(writer.buffer) == 0
    assert tmp_csvfile.exists()

    with tmp_csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3
        assert rows == [["1", "2", "3"], ["1", "2", "3"], ["4", "5", "6"]]


def test_csvwriter_flush_buffer(tmp_csvfile):
    """
    Test that the BufferedCSVWriter flushes the buffer when explicitly told to.
    """
    writer = BufferedCSVWriter(tmp_csvfile, [1,2,3], buffer_size=2)
    writer.append({1:1, 2:2, 3:3})
    writer.flush_buffer()

    assert len(writer.buffer) == 0
    assert tmp_csvfile.exists()

    with tmp_csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows == [["1", "2", "3"],["1", "2", "3"]]


def test_csvwriter_close(tmp_csvfile):
    """
    Test that the BufferedCSVWriter flushes the buffer and closes the file when closed.
    """
    writer = BufferedCSVWriter(tmp_csvfile, [1,2,3], buffer_size=2)
    writer.append({1:1, 2:2, 3:3})
    writer.close()

    assert len(writer.buffer) == 0
    assert tmp_csvfile.exists()

    with tmp_csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows == [["1", "2", "3"], ["1", "2", "3"]]

def test_csvwriter_header(tmp_csvfile):
    """
    Headers determine the order and structure of the csv
    - order cols by header not input data
    - ignore extra fields
    - '' in missing fields
    """
    writer = BufferedCSVWriter(tmp_csvfile, header=['a','b','c'], buffer_size=2)
    # out of order
    writer.append({'c': 'c', 'b': 'b', 'a': 'a'})
    # extra field
    writer.append({'a':'a', 'extra': 'extra', 'b': 'b', 'c': 'c'})
    # None for missing field
    writer.append({'a': 'a', 'c': 'c'})
    writer.close()

    expected = [
        ['a','b','c'],
        ['a','b','c'],
        ['a','b','c'],
        ['a', '', 'c']
    ]

    with tmp_csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row, ex in zip(rows, expected):
            assert row == ex



def test_read(wirefree):
    """
    Test that we can read a frame!

    For now since we're just using the example, don't try and validate the output,
    we'll do that later.
    """
    n_frames = 20

    # before we enter the context manager, we shouldn't be able to read
    with pytest.raises(RuntimeError):
        frame = wirefree.read()

    # failing to read should not increment the frame and it should still be None
    assert wirefree.frame is None

    with wirefree:
        for i in range(n_frames):
            # Frame indicates what frame we are just about to read
            assert wirefree.frame == i

            frame = wirefree.read()

            # the frame is the right shape
            assert len(frame.shape) == 2
            assert frame.shape[0] == wirefree.config.height
            assert frame.shape[1] == wirefree.config.width

            # assert they're not all zeros - ie. we read some data
            assert frame.any()

            # we should have stashed frame start positions
            # if we just read the 0th frame, we should have 2 positions
            # for the 0th and 1st frame
            assert len(wirefree.positions) == i + 2

    # after we exit the context manager, we should lose our current frame
    assert wirefree.frame is None
    # we should also not be able to read anymore
    with pytest.raises(RuntimeError):
        frame = wirefree.read()
    # and the file descriptor should also be gone
    assert wirefree._f is None
    # but we should keep our positions
    assert len(wirefree.positions) == n_frames + 1


def test_return_headers(wirefree):
    """
    We can return the headers for the individual buffers in a frame
    """
    with wirefree:
        frame_object = wirefree.read(return_header=True)
        assert isinstance(frame_object, SDCardFrame)

        assert len(frame_object.headers) == 5
        assert all([isinstance(b, SDBufferHeader) for b in frame_object.headers])


def test_frame_count(wirefree):
    """
    We can infer the total number of frames in a recording from the data header
    """
    # known max frames given the data header in the example data
    assert wirefree.frame_count == 388

    # if we try and read past the end, we get an exception
    with wirefree:
        wirefree.frame = 389
        with pytest.raises(EndOfRecordingException):
            frame = wirefree.read()


def test_relative_path():
    """
    Test that we can use both relative and absolute paths in the SD card model
    """
    # get absolute path of working directory, then get relative path to data from there
    abs_cwd = Path(os.getcwd()).resolve()
    abs_child = Path(__file__).parent.parent / "data" / "wirefree_example.img"
    rel_path = abs_child.relative_to(abs_cwd)

    assert not rel_path.is_absolute()
    sdcard = SDCard(drive=rel_path, layout="wirefree-sd-layout")

    # check we can do something basic like read config
    assert sdcard.config is not None

    # check it remains relative after init
    assert not sdcard.drive.is_absolute()

    # now try with an absolute path
    abs_path = rel_path.resolve()
    assert abs_path.is_absolute()
    sdcard_abs = SDCard(drive=abs_path, layout="wirefree-sd-layout")
    assert sdcard_abs.config is not None
    assert sdcard_abs.drive.is_absolute()


@pytest.mark.parametrize(
    ["file", "fourcc", "hash"],
    [
        ("video.avi", "GREY", "de1a5a0bd06c17588cef2130c96a883a58eeedc1b46f2b89e0233ff8c4ef4e32"),
    ],
)
def test_write_video(wirefree, file, fourcc, hash):
    """
    Test that we can write videos from an SD card!!
    """
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / file
        wirefree.to_video(path, fourcc=fourcc, progress=False)
        file_hash = hash_video(path)
        assert file_hash == hash


@pytest.mark.parametrize(
    ["n_frames", "hash"], [(50, "9b48a4ae3458187072d73840b51c9de6f986dd2f175c566dbb1d44216c313e19")]
)
def test_to_img(wirefree_battery, n_frames, hash, tmp_path):
    out_file = tmp_path / "test_toimg.img"
    wirefree_battery.to_img(out_file, n_frames, force=True)
    out_hash = hash_file(out_file)

    assert out_hash == hash

    sd = SDCard(out_file, "wirefree-sd-layout-battery")

    # we should be able to read all the frames!
    frames = []
    with sd:
        for i in range(n_frames):
            frames.append(sd.read(return_header=True))

    assert not any([f.frame is None for f in frames])
    assert all([np.nonzero(f.frame) for f in frames])

    # we should not write to file if it exists and force is False
    assert out_file.exists()
    mtime = os.path.getmtime(out_file)

    with pytest.raises(FileExistsError):
        wirefree_battery.to_img(out_file, n_frames, force=False)

    assert mtime == os.path.getmtime(out_file)

    # forcing should overwrite the file
    wirefree_battery.to_img(out_file, n_frames, force=True)
    assert mtime != os.path.getmtime(out_file)

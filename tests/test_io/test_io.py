import csv

import pytest

from mio.io import BufferedCSVWriter


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
    writer.append({1: 1, 2: 2, 3: 3})
    assert len(writer.buffer) == 2

    writer.append({1: 4, 2: 5, 3: 6})
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
    writer = BufferedCSVWriter(tmp_csvfile, [1, 2, 3], buffer_size=2)
    writer.append({1: 1, 2: 2, 3: 3})
    writer.flush_buffer()

    assert len(writer.buffer) == 0
    assert tmp_csvfile.exists()

    with tmp_csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows == [["1", "2", "3"], ["1", "2", "3"]]


def test_csvwriter_close(tmp_csvfile):
    """
    Test that the BufferedCSVWriter flushes the buffer and closes the file when closed.
    """
    writer = BufferedCSVWriter(tmp_csvfile, [1, 2, 3], buffer_size=2)
    writer.append({1: 1, 2: 2, 3: 3})
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
    writer = BufferedCSVWriter(tmp_csvfile, header=["a", "b", "c"], buffer_size=2)
    # out of order
    writer.append({"c": "c", "b": "b", "a": "a"})
    # extra field
    writer.append({"a": "a", "extra": "extra", "b": "b", "c": "c"})
    # None for missing field
    writer.append({"a": "a", "c": "c"})
    writer.close()

    expected = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"], ["a", "", "c"]]

    with tmp_csvfile.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row, ex in zip(rows, expected):
            assert row == ex

import re
from pathlib import Path

import multiprocessing
import numpy as np
import os
import pytest
import pandas as pd
import sys
import signal
import time
from contextlib import contextmanager
from bitstring import BitArray, Bits
from typing import Generator
import warnings
import cv2

from mio.const import BASE_DIR
from mio.stream_daq import StreamDevConfig, StreamDaq, iter_buffers
from mio.models.process import FreqencyMaskingConfig
from mio.utils import hash_video, hash_file
from mio.io import VideoWriter
from .conftest import DATA_DIR, CONFIG_DIR


@pytest.fixture(params=[pytest.param(5, id="buffer-size-5"), pytest.param(10, id="buffer-size-10")])
def default_streamdaq(set_okdev_input, request) -> StreamDaq:

    daqConfig = StreamDevConfig.from_id("test-wireless-200px")
    daqConfig.runtime.frame_buffer_queue_size = request.param
    daqConfig.runtime.image_buffer_queue_size = request.param
    daqConfig.runtime.serial_buffer_queue_size = request.param

    data_file = DATA_DIR / "stream_daq_test_fpga_raw_input_200px.bin"
    set_okdev_input(data_file)

    daq_inst = StreamDaq(device_config=daqConfig)
    return daq_inst


# Second parameter makes sure the filtering does not affect the video output
@pytest.mark.parametrize("buffer_size", [5, 50])
@pytest.mark.parametrize(
    "device_config,filter_config,data,video_hash_list,show_video",
    [
        (
            "test-wireless-200px",
            None,
            "stream_daq_test_fpga_raw_input_200px.bin",
            [
                "22fa7249faffff45f5f5aa12d36399da9e8be2f7c578ca2a3c7dccbaddc9063e",
            ],
            False,
        ),
        (
            "test-wireless-200px",
            "test_remove_stripe_example",
            "stream_daq_test_fpga_raw_input_200px.bin",
            [
                "22fa7249faffff45f5f5aa12d36399da9e8be2f7c578ca2a3c7dccbaddc9063e",
            ],
            False,
        ),
    ],
)
def test_video_output(
    device_config,
    filter_config,
    data,
    video_hash_list,
    tmp_path,
    show_video,
    set_okdev_input,
    buffer_size,
):
    output_video = tmp_path / "output.avi"
    output_csv = tmp_path / "output.csv"

    daqConfig = StreamDevConfig.from_id(device_config)
    daqConfig.runtime.frame_buffer_queue_size = buffer_size
    daqConfig.runtime.image_buffer_queue_size = buffer_size
    daqConfig.runtime.serial_buffer_queue_size = buffer_size

    if filter_config:
        processor_for_visualization = FreqencyMaskingConfig.from_id(filter_config)
    else:
        processor_for_visualization = None

    data_file = DATA_DIR / data
    set_okdev_input(data_file)

    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst.capture(
        source="fpga",
        video=output_video,
        metadata=output_csv,
        show_video=show_video,
        freq_mask_config=processor_for_visualization,
    )

    assert output_video.exists()
    assert output_csv.exists()

    output_video_hash = hash_video(output_video)
    assert output_video_hash in video_hash_list

    # Regression test: metadata indices must align with AVI frames
    df = pd.read_csv(output_csv)

    cap = cv2.VideoCapture(str(output_video))
    avi_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    valid_indices = df[df["reconstructed_frame_index"] != -1]["reconstructed_frame_index"]
    expected_max_index = avi_frame_count - 1
    actual_max_index = int(valid_indices.max()) if len(valid_indices) > 0 else -1

    # Max index should match AVI frame count
    assert (
        actual_max_index == expected_max_index
    ), f"Max index {actual_max_index} != AVI max {expected_max_index}"


@pytest.mark.parametrize(
    "config,data",
    [
        (
            "test-wireless-200px",
            "stream_daq_test_fpga_raw_input_200px.bin",
        )
    ],
)
def test_binary_output(config, data, set_okdev_input, tmp_path):
    daqConfig = StreamDevConfig.from_id(config)

    data_file = DATA_DIR / data
    set_okdev_input(data_file)

    output_file = tmp_path / "output.bin"

    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst.capture(source="fpga", binary=output_file, show_video=False)

    assert output_file.exists()

    assert hash_file(data_file) == hash_file(output_file)


@pytest.mark.parametrize("write_metadata", [True, False])
def test_csv_output(tmp_path, default_streamdaq, write_metadata, caplog):
    """
    Giving a path to the ``metadata`` capture kwarg should save header metadata to a csv
    """
    output_csv = tmp_path / "output.csv"

    if write_metadata:
        default_streamdaq.capture(source="fpga", metadata=output_csv, show_video=False)

        df = pd.read_csv(output_csv)
        # actually not sure what we should be looking for here, for now we just check for shape
        # this should be the same as long as the test data stays the same,
        # but it's a pretty weak test.
        assert df.shape == (907, 15)

        # the underlying csv should have the same number of column headers as data columsn
        # if there is a mismatch, the index will turn into a multi-index
        assert isinstance(df.index, pd.RangeIndex)

        # we should have the same columns in the same order as our header format plus reconstruction metadata
        col_names = df.columns.to_list()
        expected = default_streamdaq.header_fmt.model_dump(
            exclude_none=True, exclude=set(default_streamdaq.header_fmt.HEADER_FIELDS)
        )
        expected = [h[0] for h in sorted(expected.items(), key=lambda x: x[1])]

        # Add runtime_metadata fields
        from mio.models.stream import RuntimeMetadata

        runtime_fields = list(RuntimeMetadata.model_fields.keys())
        expected.extend(runtime_fields)
        assert col_names == expected

        # ensure there were no errors during capture
        for record in caplog.records:
            assert "Exception saving headers" not in record.msg

        # ensure the buffer_recv_index increments from 0 to the number of buffers in the data file as an array
        buffer_recv_index = df.buffer_recv_index.to_numpy()
        assert np.all(buffer_recv_index == np.arange(0, len(buffer_recv_index)))

        # ensure the reconstructed frame index array that excludes -1 monotonically increases from 0
        reconstructed_frame_index = df.reconstructed_frame_index.to_numpy()
        reconstructed_frame_index = reconstructed_frame_index[reconstructed_frame_index != -1]
        assert np.all(np.diff(reconstructed_frame_index) >= 0)

        # ensure that reconstructed frame index contains all values from 0 to the number of frames in the data file
        unique_reconstructed_frame_index = np.unique(reconstructed_frame_index)
        assert np.all(
            unique_reconstructed_frame_index == np.arange(0, len(unique_reconstructed_frame_index))
        )

    else:
        default_streamdaq.capture(source="fpga", metadata=None, show_video=False)
        assert not output_csv.exists()


def test_processing_speed(tmp_path, default_streamdaq):
    """
    Processing speed test of the stream daq.
    For being generous for runs in CI, the test will pass if the processing speed is faster than the test_fail_fps.
    This will output a warning if it is slower than the warning_fps.
    """
    test_fail_fps = 10
    warning_fps = 40
    output_csv = tmp_path / "output.csv"

    default_streamdaq.capture(source="fpga", metadata=output_csv, show_video=False)

    df = pd.read_csv(output_csv)

    unix_time_first = df.iloc[0]["buffer_recv_unix_time"]
    unix_time_last = df.iloc[-1]["buffer_recv_unix_time"]
    time_taken = unix_time_last - unix_time_first

    frame_index_first = df.iloc[0]["frame_num"]
    frame_index_last = df.iloc[-1]["frame_num"]
    num_frames = frame_index_last - frame_index_first

    processing_fps = num_frames / time_taken

    if processing_fps < warning_fps:
        warnings.warn(
            f"Processing speed is {processing_fps} FPS, which is slower than the required {warning_fps} FPS"
        )

    assert processing_fps > test_fail_fps


def test_csv_no_duplicates(tmp_path, set_okdev_input):
    """
    Regression test for a bug where header rows would be written multiple times when
    buffer_npix was miscalculated and the buffer_list wasn't cleared after being put in the
    queue multiple times.
    """
    bad_buffer_npix = [5072, 5072, 5072, 5072]
    output_csv = tmp_path / "output.csv"

    daqConfig = StreamDevConfig.from_id("test-wireless-200px")

    data_file = DATA_DIR / "stream_daq_test_fpga_raw_input_200px.bin"
    set_okdev_input(data_file)

    daq_inst = StreamDaq(device_config=daqConfig)
    daq_inst._buffer_npix = bad_buffer_npix

    assert daq_inst.buffer_npix == bad_buffer_npix
    daq_inst.capture(source="fpga", metadata=output_csv, show_video=False)
    assert daq_inst.buffer_npix == bad_buffer_npix
    df = pd.read_csv(output_csv)
    vals, counts = np.unique(df.buffer_count, return_counts=True)
    assert all(
        [c == 1 for c in counts]
    ), "Duplicated buffer indexes found, rows being written twice"


# This is a helper function for test_continuous_and_termination() that is currently skipped
"""
def capture_wrapper(default_streamdaq, source, show_video, continuous):
    try:
        default_streamdaq.capture(source=source, show_video=show_video, continuous=continuous)
    except KeyboardInterrupt:
        pass # expected
"""


@pytest.mark.skip(
    "Needs to be implemented. Temporary skipped because tests fail in some OS (See GH actions)."
)
@pytest.mark.timeout(10)
def test_continuous_and_termination(tmp_path, default_streamdaq):
    """
    Make sure continuous mode runs forever until interrupted, and that all processes are
    cleaned up when the capture process is terminated.
    """
    """
    timeout = 1

    capture_process = multiprocessing.Process(target=capture_wrapper, args=(default_streamdaq, "fpga", False, True))

    capture_process.start()
    alive_processes = default_streamdaq.alive_processes()
    initial_alive_processes = len(alive_processes)
    
    time.sleep(timeout)

    alive_processes = default_streamdaq.alive_processes()
    assert len(alive_processes) == initial_alive_processes
    
    os.kill(capture_process.pid, signal.SIGINT)
    capture_process.join()

    alive_processes = default_streamdaq.alive_processes()
    assert len(alive_processes) == 0
    """
    pass


def test_metadata_plotting(tmp_path, default_streamdaq):
    """
    Setting the capture kwarg ``show_metadata == True`` should plot the frame metadata
    during capture.
    """
    default_streamdaq.capture(source="fpga", show_metadata=True, show_video=False)

    # unit tests for the stream plotter should go elsewhere, here we just
    # test that the object was instantiated and that it got the data it should have
    assert default_streamdaq._header_plotter is not None
    assert [
        k for k in default_streamdaq._header_plotter.data.keys()
    ] == default_streamdaq.config.runtime.plot.keys
    assert all(
        [
            len(v) == default_streamdaq.config.runtime.plot.history
            for v in default_streamdaq._header_plotter.data.values()
        ]
    )
    assert (
        len(default_streamdaq._header_plotter.index)
        == default_streamdaq.config.runtime.plot.history
    )


def test_bitfile_names():
    """
    Bitfile names should have no periods or whitespace in the filenames (except for the .bit extension)
    """
    pattern = re.compile(r"\.(?!bit$)|\s")
    for path in Path(BASE_DIR).glob("**/*.bit"):
        assert not pattern.search(str(path.name))


@pytest.mark.parametrize("read_size", [3, 5, 7])
def test_iter_buffers(read_size: int, tmp_path: Path):
    """
    iter_buffers should accept an iterator that yield bytes,
    and split it by the preamble in a way that's insensitive to
    the length of the read size
    """
    preamble_bytes = b"ab"
    n_reps = 3

    preamble = Bits(preamble_bytes)
    buffer = preamble_bytes + b"000"
    buffer_rep = buffer * n_reps

    def _iterator(read_size: int) -> Generator[bytes, None, None]:
        nonlocal buffer_rep
        for i in range(0, len(buffer_rep), read_size):
            yield buffer_rep[i : i + read_size]

    got_buffers = []
    for buf in iter_buffers(_iterator(read_size), preamble=preamble):
        got_buffers.append(buf)

    assert all([buf == buffer for buf in got_buffers])


def test_writer_returns_match_avi_frame_count(tmp_path: Path, set_okdev_input, monkeypatch):
    """
    Count write_frame calls and True returns from VideoWriter and compare the number of True returns against the number of frames reported in the AVI.
    """
    call_count = {"calls": 0, "ok": 0, "failed": 0}
    original = VideoWriter.write_frame

    def wrapped(self, frame):  # type: ignore[no-redef]
        call_count["calls"] += 1
        ok = original(self, frame)
        if ok:
            call_count["ok"] += 1
        else:
            call_count["failed"] += 1  # This shouldn't happen but just in case
        return ok

    monkeypatch.setattr(VideoWriter, "write_frame", wrapped, raising=True)

    output_video = tmp_path / "out.avi"

    daqConfig = StreamDevConfig.from_id("test-wireless-200px")
    data_file = DATA_DIR / "stream_daq_test_fpga_raw_input_200px.bin"
    set_okdev_input(data_file)

    daq = StreamDaq(device_config=daqConfig)
    daq.capture(source="fpga", video=output_video, metadata=None, show_video=False)

    assert output_video.exists()

    cap = cv2.VideoCapture(str(output_video))
    avi_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # No false returns
    assert call_count["failed"] == 0, f"write_frame False returns ({call_count['failed']})"

    # Successes should equal the frames counted by AVI?
    assert (
        call_count["ok"] == avi_frames
    ), f"write_frame True returns ({call_count['ok']}) != AVI frames ({avi_frames})"

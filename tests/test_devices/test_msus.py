from collections import defaultdict

import numpy as np
import pytest

from mio.devices.stream import StreamDevConfig


@pytest.mark.parametrize(
    "thresh_low,thresh_high",
    [(300, 900)],
)
def test_pixels_with_known_input(msus_pixel_buffers, thresh_low, thresh_high):
    """
    we correctly recover the pixel values from msus encoded data.

    We use a raw sample from the device where the sensor is xposed to bright light
    for the first few frames, and then covered in the last few
    to generate "known input,"
    since the device is not capable of generating a test pattern.

    This test does not test the general correctness of frame formatting,
    like its error handling, correctness of shape, etc.
    Here we are just testing the *values* of the frames - whether we get
    correct pixel values (or as close as we can verify with such a coarse notion of known input)
    """

    frame_buffers = defaultdict(list)

    # collect pixel buffers by frame
    for header, pixels in msus_pixel_buffers:
        frame_buffers[header.frame_num].append(pixels)

    # delete the first and last, we assume they are incomplete
    del frame_buffers[min(frame_buffers.keys())]
    del frame_buffers[max(frame_buffers.keys())]

    frames = []
    for frame_n in sorted(frame_buffers.keys()):
        frames.append(np.concat(frame_buffers[frame_n]).flatten())

    # first frames should be bright, last frames should be dark
    # this should be stricter, but the input data is not very clean
    # (dark is not very dark)
    # so we use 75% quantile for bright, and median for dark
    bright_qts = np.quantile(frames[0], (0.25, 0.5, 0.75))
    dark_qts = np.quantile(frames[-1], (0.25, 0.5, 0.75))

    # 75% of pixels are brighter than high thresh, vice versa for low
    assert bright_qts[0] > thresh_high
    assert dark_qts[1] < thresh_low


def test_buffer_npix(msus_pixel_buffers):
    """Our trimming and padding is correctly derived from the config"""
    frame_buffers = defaultdict(list)

    config = StreamDevConfig.from_id("MSUS")

    for header, pixels in msus_pixel_buffers:
        frame_buffers[header.frame_num].append(pixels)

    # discard first and last which may be incomplete in the sample
    frames = list(frame_buffers.keys())
    del frame_buffers[frames[0]]
    del frame_buffers[frames[-1]]

    for pixels in frame_buffers.values():
        pixel_lengths = [len(p) for p in pixels]
        assert pixel_lengths == config.buffer_npix

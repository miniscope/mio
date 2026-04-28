"""
Test for skvideo.io.FFmpegWriter.
"""

import cv2
import numpy as np
from skvideo.io import FFmpegWriter


def test_write_video(tmp_path):
    """
    Write a random grayscale video and check the number of frames.
    Check with OpenCV that the video is written correctly.
    """

    width, height, num_frames, fps = 200, 200, 200, 20
    input_dict = {"-framerate": str(fps)}
    output_dict = {
        "-vcodec": "rawvideo",
        "-f": "avi",
        "-pix_fmt": "gray",
        "-vsync": "0",  # probably not necessary but for safety
    }

    out = tmp_path / "test.avi"
    writer = FFmpegWriter(str(out), inputdict=input_dict, outputdict=output_dict)
    for _ in range(num_frames):
        frame = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        writer.writeFrame(frame)
    writer.close()

    cap = cv2.VideoCapture(str(out))
    avi_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert avi_frames == num_frames

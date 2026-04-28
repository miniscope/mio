import sys
from enum import Enum
from pprint import pformat

import cv2
import numpy as np
import pytest
import yaml
from pydantic import BaseModel

from mio.models.process import BlackAreaDetectorConfig, DenoiseConfig, NoisePatchConfig
from mio.process.frame_helper import BlackAreaDetector, InvalidFrameDetector

from ..conftest import DATA_DIR

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class GroundTruthCategory(str, Enum):
    check_pattern = "check_pattern"
    blacked_out = "blacked_out"


class GroundTruthSubcategories(TypedDict, total=False):
    less_than_one_row: list[int]
    several_rows: list[int]
    one_block: list[int]
    several_blocks: list[int]
    majority_of_frame: list[int]


class NoiseGroundTruth(BaseModel):
    frames: dict[GroundTruthCategory, GroundTruthSubcategories]


@pytest.mark.parametrize(
    "video,ground_truth",
    [
        (str(DATA_DIR / "wireless_corrupted.avi"), str(DATA_DIR / "wireless_corrupted.yaml")),
        (
            str(DATA_DIR / "wireless_corrupted_extended.mp4"),
            str(DATA_DIR / "wireless_corrupted_extended.yaml"),
        ),
    ],
)
@pytest.mark.parametrize(
    "noise_detection_method,noise_category",
    [
        (["gradient"], GroundTruthCategory.check_pattern),
        (["black_area"], GroundTruthCategory.blacked_out),
    ],
)
def test_noisy_frame_detection(video, ground_truth, noise_detection_method, noise_category):
    """
    Contrast method of noise detection should correctly label frames corrupted
    by speckled noise
    """
    if "gradient" in noise_detection_method or "black_area" in noise_detection_method:
        global_config: DenoiseConfig = DenoiseConfig.from_id("denoise_noise_detection_test")
    else:
        raise ValueError("Invalid noise detection method")

    config: NoisePatchConfig = global_config.noise_patch
    config.method = noise_detection_method

    with open(ground_truth) as yfile:
        expected = NoiseGroundTruth(**yaml.safe_load(yfile))

    if noise_category not in expected.frames:
        pytest.skip(f"No frames with noise category {noise_category} in ground truth")

    video = cv2.VideoCapture(video)

    detector = InvalidFrameDetector(noise_patch_config=config)

    detected_frames = []
    previous_frame = None
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_frame is None:
            previous_frame = frame

        is_noisy, mask = detector.find_invalid_area(frame=frame)
        if not is_noisy:
            previous_frame = frame

        detected_frames.append(is_noisy)

    detected_frame_indices = np.where(detected_frames)[0].tolist()

    # Detect false negatives
    missed = []
    for subcategory, expected_frames in expected.frames[noise_category].items():
        expected_set = set(expected_frames)
        detected_set = set(detected_frame_indices)
        missed_frames = expected_set - detected_set  # Frames in expected but not detected
        if missed_frames:
            missed.append({"missed": missed_frames, "subcategory": subcategory})
    assert missed == [], f"Missed frames with method {noise_detection_method}: {pformat(missed)}"

    # Detect false positives
    # False positives are across all noise categories, regardless of what we're testing here.
    all_expected = set().union(
        *[set(frame) for subcategory in expected.frames.values() for frame in subcategory.values()]
    )
    extra_frames = set(detected_frame_indices) - all_expected
    assert extra_frames == set(), f"Detected extra, non-noise frames as noisy: {extra_frames}"


@pytest.mark.parametrize(
    "min_rows,expected_noisy",
    [
        (1, True),  # default: any flagged row triggers detection
        (5, True),  # exactly 5 noisy rows meets the threshold
        (10, False),  # only 5 noisy rows, below threshold of 10
    ],
)
def test_black_area_min_rows(min_rows, expected_noisy):
    """min_rows controls how many flagged rows are needed to mark a frame as invalid."""
    # Create a 50x50 frame with 5 rows of consecutive zeros (noisy) and the rest bright
    frame = np.ones((50, 50), dtype=np.uint8) * 128
    for row in range(5):
        frame[row, :30] = 0  # 30 consecutive black pixels in rows 0-4

    config = BlackAreaDetectorConfig(
        consecutive_threshold=10,
        value_threshold=0,
        min_rows=min_rows,
    )
    detector = BlackAreaDetector(config)
    is_noisy, mask = detector.find_invalid_area(frame)
    assert (
        is_noisy == expected_noisy
    ), f"min_rows={min_rows}: expected noisy={expected_noisy}, got {is_noisy}"

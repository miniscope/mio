"""
These tests don't really test much, because this module basically just wraps the actual
processors which are defined in process.frame_helper (and the tests are in test_frame_helper)

This module needs a decent dose of abstraction, so not spending time exhaustively testing
each of the methods.
"""

import numpy as np
import pytest

from mio.process.video import (
    NoisePatchProcessor,
    FreqencyMaskProcessor,
    PassThroughProcessor,
    MinProjSubtractProcessor,
    MinimumProjectionConfig,
)
from mio.models.process import DenoiseConfig


@pytest.fixture()
def video_frame() -> np.ndarray:
    """idk it's a frame of all 128s ig lol"""
    return np.ones((100, 100), dtype=np.uint8) * 128


@pytest.fixture()
def random_8bit_video_frame() -> np.ndarray:
    """Frame with pseudo-random 8-bit values."""
    seed = 42  # Arbitrary fixed seed to make the output deterministic
    np.random.seed(seed)
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


def test_noise_patch_processor(video_frame, tmp_path):
    denoise_config = DenoiseConfig.from_id("denoise_example")
    denoise_config.noise_patch.enable = True
    denoise_config.noise_patch.output_result = True

    processor = NoisePatchProcessor("denoise_example", denoise_config.noise_patch, tmp_path)
    processed_frame = processor.process_frame(video_frame, 0)

    assert isinstance(processed_frame, np.ndarray)
    assert processor.name == "denoise_example"
    assert processor.output_enable


def test_noise_patch_processor_no_config(random_8bit_video_frame, tmp_path):
    denoise_config = DenoiseConfig.from_id("denoise_example")
    denoise_config.noise_patch.enable = True
    denoise_config.noise_patch.gradient_config = None
    denoise_config.noise_patch.black_area_config = None

    # This should raise a ValueError because the necessary configs are not provided
    with pytest.raises(ValueError):
        NoisePatchProcessor("denoise_example", denoise_config.noise_patch, tmp_path)


def test_freqency_mask_processor(video_frame, tmp_path):
    denoise_config = DenoiseConfig.from_id("denoise_example")
    denoise_config.frequency_masking.enable = True
    denoise_config.frequency_masking.output_result = True

    processor = FreqencyMaskProcessor(
        "test_freq_mask",
        denoise_config.frequency_masking,
        100,
        100,
        tmp_path,
    )
    processed_frame = processor.process_frame(video_frame, 0)

    assert isinstance(processed_frame, np.ndarray)
    assert processor.name == "test_freq_mask"
    assert processor.output_enable


def test_pass_through_processor(video_frame, tmp_path):
    processor = PassThroughProcessor("test_pass_through", tmp_path)
    processed_frame = processor.process_frame(video_frame, 0)

    assert isinstance(processed_frame, np.ndarray)
    assert np.array_equal(processed_frame, video_frame)
    assert processor.name == "test_pass_through"


def test_min_proj_subtract_processor(video_frame, tmp_path):
    video_frames = [video_frame for _ in range(10)]
    min_proj_config = MinimumProjectionConfig()
    min_proj_config.output_result = True

    processor = MinProjSubtractProcessor("test_min_proj", min_proj_config, tmp_path, video_frames)
    processor.normalize_stack()

    assert processor.name == "test_min_proj"
    assert processor.output_enable
    assert all(isinstance(frame, np.ndarray) for frame in processor.output_frames)

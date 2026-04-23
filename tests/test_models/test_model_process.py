from pydantic import ValidationError
import pytest

from mio.models.process import DenoiseConfig
from mio.const import BASE_DIR
from mio.utils import hash_video, hash_file
from ..conftest import DATA_DIR, CONFIG_DIR


@pytest.mark.skip(reason="Not sure how to get config.")
def test_interactive_display_config():
    config = DenoiseConfig.from_id("denoise_test").interactive_display
    assert config.enable == True
    assert config.start_frame == 40
    assert config.end_frame == 140


@pytest.mark.skip(reason="Not sure how to get config.")
def test_minimum_projection_config():
    config = DenoiseConfig.from_id("denoise_test").minimum_projection
    assert config.enable == True
    assert config.normalize == True
    assert config.output_result == False
    assert config.output_min_projection == False


@pytest.mark.skip(reason="Not sure how to get config.")
def test_noise_patch_config():
    config = DenoiseConfig.from_id("denoise_test").noise_patch
    assert config.enable == True
    assert config.output_result == True


@pytest.mark.skip(reason="Not sure how to get config.")
def test_freqency_masking_config():
    config = DenoiseConfig.from_id("denoise_test").frequency_masking
    assert config.enable == True


@pytest.mark.skip(reason="Not sure how to get config.")
def test_denoise_config():
    config = DenoiseConfig.from_id("denoise_test")
    assert config.noise_patch.enable == True
    assert config.frequency_masking.enable == True

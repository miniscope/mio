"""
Module for preprocessing data.
"""

from typing import Literal

from pydantic import BaseModel, Field

from mio.models import MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin
from mio.models.stream import StreamDevConfig


class MinimumProjectionConfig(BaseModel):
    """
    Configuration for calculating and processing the video based on minimum projection of the stack.
    This is used to acquire the minimum intensity projection (static background) of the video,
    and normalize the video based on the minimum projection.
    """

    enable: bool = Field(
        default=True,
        description="Enable minimum projection.",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize the video using minimum projection."
        "If True, the video will be normalized using the minimum projection,"
        "so that the minimum value is 0 and the maximum value is the maximum of uint8.",
    )
    output_result: bool = Field(
        default=False,
        description="Output the normalized video stream.",
    )
    output_min_projection: bool = Field(
        default=False,
        description="Output the minimum projection frame.",
    )


class MSEDetectorConfig(BaseModel):
    """
    Configraiton for detecting invalid frames based on mean squared error.
    """

    threshold: float = Field(
        ...,
        description="Threshold for detecting invalid frames based on mean squared error.",
    )
    device_config_id: str | None = Field(
        default=None,
        description="ID of the stream device configuration used for aquiring the video."
        "This is used in the mean_error method to compare frames"
        " in the units of data transfer buffers.",
    )
    buffer_split: int = Field(
        default=1,
        description="Number of splits to make in the buffer when detecting noisy areas."
        "This further splits the buffer into smaller patches to detect small noisy areas."
        "This is used in the mean_error method.",
    )
    diff_multiply: int = Field(
        default=1,
        description="Multiplier for visualizing the diff between the current and previous frame.",
    )

    _device_config: StreamDevConfig | None = None

    @property
    def device_config(self) -> StreamDevConfig:
        """
        Get the device configuration based on the device_config_id.
        This is used in the mean_error method to compare frames in the units of data buffers.
        """
        if self._device_config is None:
            self._device_config = StreamDevConfig.from_any(self.device_config_id)
        return self._device_config


class GradientDetectorConfig(BaseModel):
    """
    Configraiton for detecting invalid frames based on gradient.
    """

    threshold: float = Field(
        ...,
        description="Threshold for detecting invalid frames based on gradient.",
    )


class BlackAreaDetectorConfig(BaseModel):
    """
    Configraiton for detecting invalid frames based on black area.
    """

    consecutive_threshold: int = Field(
        default=5,
        description="Number of consecutive black pixels required to classify a row as noisy.",
    )
    value_threshold: int = Field(
        default=0,
        description="Pixel intensity value below which a pixel is considered 'black'.",
    )
    min_rows: int = Field(
        default=1,
        description="Minimum number of flagged rows required to mark the frame as invalid. "
        "Default of 1 preserves original behavior. For calcium imaging, values around 10 "
        "reduce false positives from naturally dark regions.",
    )


class NoisePatchConfig(BaseModel):
    """
    Configuration for patch based noise handling.
    This is used to detect noisy areas in each frame and drop the frame if it is noisy.
    """

    enable: bool = Field(
        default=True,
        description="Enable patch based noise handling.",
    )
    method: list[Literal["mean_error", "gradient", "black_area"]] = Field(
        default="gradient",
        description="Method for detecting noise."
        "gradient: Detection based on the gradient of the frame row."
        "mean_error: Detection based on the mean error with the same row of the previous frame."
        "black_area: Detection based on the number of consecutive black pixels in a row.",
    )
    mean_error_config: MSEDetectorConfig | None = Field(
        default=None,
        description="Configuration for detecting invalid frames based on mean squared error."
        " Any positive value or zero is valid.",
    )
    gradient_config: GradientDetectorConfig | None = Field(
        default=None,
        description="Configuration for detecting invalid frames based on gradient.",
    )
    black_area_config: BlackAreaDetectorConfig | None = Field(
        default=None,
        description="Configuration for detecting invalid frames based on black area.",
    )
    output_result: bool = Field(
        default=False,
        description="Output the output video stream.",
    )
    output_noise_patch: bool = Field(
        default=False,
        description="Output the noise patch video"
        "This highlights the noisy areas found in the video stream.",
    )
    output_diff: bool = Field(
        default=False,
        description="Output the diff video stream."
        "The diff video stream shows the difference between the current and previous frame."
        "This is used in the mean_error method.",
    )
    output_noisy_frames: bool = Field(
        default=True,
        description="Output the stack of noisy frames as an independent video stream.",
    )


class FrequencyMaskingConfig(MiniscopeConfig, ConfigYAMLMixin):
    """
    Configuration for frequency filtering.
    This includes a spatial low-pass filter and vertical and horizontal band elimination filters.
    """

    enable: bool = Field(
        default=True,
        description="Enable frequency filtering.",
    )
    cast_float32: bool = Field(
        default=False,
        description="Cast the input video stream to float32 before processing."
        "This is probably unnecessary and could be removed in the future.",
    )
    spatial_LPF_cutoff_radius: int = Field(
        default=...,
        description="Radius for the spatial low pass filter cutoff in pixels.",
    )
    vertical_BEF_cutoff: int = Field(
        default=5,
        description="Cutoff for the vertical band elimination filter in pixels.",
    )
    horizontal_BEF_cutoff: int = Field(
        default=0,
        description="Cutoff for the horizontal band elimination filter in pixels.",
    )
    output_result: bool = Field(
        default=False,
        description="Output the result video stream.",
    )
    output_mask: bool = Field(
        default=False,
        description="Output the mask frame image.",
    )
    output_freq_domain: bool = Field(
        default=False,
        description="Output the freq domain of the input video stream.",
    )


class InteractiveDisplayConfig(BaseModel):
    """
    Configuration for interactively displaying the video.
    This can not display long video streams efficienty and is for debugging purposes.
    """

    show_videos: bool = Field(
        default=False,
        description="Enable interactive display.",
    )
    start_frame: int | None = Field(
        default=...,
        description="Frame to start interactive display at.",
    )
    end_frame: int | None = Field(
        default=...,
        description="Frame to end interactive display at.",
    )
    display_freq_mask: bool = Field(
        default=False,
        description="Interactively display the mask before starting processing",
    )


class DenoiseConfig(MiniscopeConfig, ConfigYAMLMixin):
    """
    Configuration for denoising a video.
    """

    interactive_display: InteractiveDisplayConfig | None = Field(
        default=None,
        description="Configuration for interactively displaying the video.",
    )
    noise_patch: NoisePatchConfig | None = Field(
        default=None,
        description="Configuration for patch based noise handling.",
    )
    frequency_masking: FrequencyMaskingConfig | None = Field(
        default=None,
        description="Configuration for frequency masking.",
    )
    end_frame: int | None = Field(
        default=None,
        description="Frame to end processing at. If None, process until the end of the video.",
    )
    minimum_projection: MinimumProjectionConfig | None = Field(
        default=None,
        description="Configuration for processing based on minimum projection.",
    )
    output_result: bool = Field(
        default=True,
        description="Output the result video stream.",
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory to save the output video streams and frames.",
    )

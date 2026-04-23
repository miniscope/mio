"""
This module contains a helper class for frame operations.
"""

from abc import abstractmethod

import cv2
import numpy as np

from mio import init_logger
from mio.models.process import (
    BlackAreaDetectorConfig,
    FrequencyMaskingConfig,
    GradientDetectorConfig,
    MSEDetectorConfig,
    NoisePatchConfig,
)

logger = init_logger("frame_helper")


class BaseSingleFrameHelper:
    """
    Base class for single frame operations.
    """

    def __init__(self):
        """
        Initialize the BaseSingleFrameHelper object.

        Returns:
            BaseSingleFrameHelper: A BaseSingleFrameHelper object.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            np.ndarray: The processed frame.
        """
        pass

    @abstractmethod
    def find_invalid_area(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Find the invalid area in a single frame.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is invalid
            and the processed frame.
        """
        pass


class InvalidFrameDetector(BaseSingleFrameHelper):
    """
    Helper class for combined invalid frame detection.
    """

    def __init__(self, noise_patch_config: NoisePatchConfig):
        """
        Initialize the FrameProcessor object.
        Block size/buffer size will be set by dev config later.

        Returns:
            NoiseDetectionHelper: A NoiseDetectionHelper object
        """
        self.config = noise_patch_config
        if noise_patch_config.method is None:
            raise ValueError("No noise detection methods provided")
        self.methods = noise_patch_config.method

        if "mean_error" in self.methods:
            if noise_patch_config.mean_error_config is None:
                raise ValueError("Mean error config must be provided for mean error detection")
            self.mse_detector = MSENoiseDetector(noise_patch_config.mean_error_config)
        if "gradient" in self.methods:
            if noise_patch_config.gradient_config is None:
                raise ValueError("Gradient config must be provided for gradient detection")
            self.gradient_detector = GradientNoiseDetector(noise_patch_config.gradient_config)
        if "black_area" in self.methods:
            if noise_patch_config.black_area_config is None:
                raise ValueError("Black area config must be provided for black area detection")
            self.black_detector = BlackAreaDetector(noise_patch_config.black_area_config)

    def find_invalid_area(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating invalid frames and the invalid area.
        """
        noisy_flag = False
        combined_noisy_area = np.zeros_like(frame, dtype=np.uint8)

        if "mean_error" in self.methods:
            noisy, noisy_area = self.mse_detector.find_invalid_area(frame)
            combined_noisy_area = np.maximum(combined_noisy_area, noisy_area)
            noisy_flag = noisy_flag or noisy

        if "gradient" in self.methods:
            noisy, noisy_area = self.gradient_detector.find_invalid_area(frame)
            combined_noisy_area = np.maximum(combined_noisy_area, noisy_area)
            noisy_flag = noisy_flag or noisy

        if "black_area" in self.methods:
            noisy, noisy_area = self.black_detector.find_invalid_area(frame)
            combined_noisy_area = np.maximum(combined_noisy_area, noisy_area)
            noisy_flag = noisy_flag or noisy

        return noisy_flag, combined_noisy_area


class GradientNoiseDetector(BaseSingleFrameHelper):
    """
    Helper class for gradient noise detection.
    """

    def __init__(self, config: GradientDetectorConfig):
        """
        Initialize the GradientNoiseDetectionHelper object.

        Parameters:
            threshold (float): The threshold for noise detection.

        Returns:
            GradientNoiseDetectionHelper: A GradientNoiseDetectionHelper object.
        """
        self.config = config

    def find_invalid_area(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        noisy, mask = self._detect_with_gradient(frame)
        return noisy, mask

    def _detect_with_gradient(
        self,
        current_frame: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        """
        Detect noise using local contrast (second derivative) in the x-dimension
        (along rows, across columns)

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        noisy_mask = np.zeros_like(current_frame, dtype=np.uint8)

        diff_x = np.diff(current_frame.astype(np.int16), n=2, axis=1)
        mean_second_diff = np.abs(diff_x).mean(axis=1)
        noisy_mask[mean_second_diff > self.config.threshold, :] = 1
        logger.debug("Row-wise means of second derivative: %s", mean_second_diff)

        # Determine if the frame is noisy (if any rows are marked as noisy)
        frame_is_noisy = noisy_mask.any()

        return frame_is_noisy, noisy_mask


class BlackAreaDetector(BaseSingleFrameHelper):
    """
    Helper class for black area detection.
    """

    def __init__(self, config: BlackAreaDetectorConfig):
        """
        Initialize the BlackAreaDetectionHelper object.

        Parameters:
            threshold (float): The threshold for noise detection.

        Returns:
            BlackAreaDetectionHelper: A BlackAreaDetectionHelper object.
        """
        self.config = config

    def find_invalid_area(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        noisy, mask = self._detect_black_pixels(frame)
        return noisy, mask

    def _detect_black_pixels(
        self,
        current_frame: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        """
        Detect black-out noise by checking for consecutive black pixels per row.

        Uses vectorized numpy (cumulative sum sliding window) instead of
        pixel-by-pixel Python loops for ~100x speedup.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is corrupted and noise mask.
        """
        consecutive_threshold = self.config.consecutive_threshold
        black_pixel_value_threshold = self.config.value_threshold

        # Boolean mask of "black" pixels, then cumsum-based sliding window
        black_mask = (current_frame <= black_pixel_value_threshold).astype(np.float32)
        cs = np.cumsum(black_mask, axis=1)

        if current_frame.shape[1] >= consecutive_threshold:
            # Sliding window: sum of `consecutive_threshold` consecutive pixels
            run_sum = cs[:, consecutive_threshold:] - cs[:, :-consecutive_threshold]
            # A row has a run if any window sums to exactly consecutive_threshold
            # (meaning all pixels in that window were black)
            row_has_run = np.any(run_sum >= consecutive_threshold, axis=1)
        else:
            row_has_run = np.zeros(current_frame.shape[0], dtype=bool)

        noisy_mask = np.zeros_like(current_frame, dtype=np.uint8)
        noisy_mask[row_has_run, :] = 1

        noisy_row_count = int(np.sum(row_has_run))
        frame_is_noisy = noisy_row_count >= self.config.min_rows
        return frame_is_noisy, noisy_mask


class MSENoiseDetector(BaseSingleFrameHelper):
    """
    Helper class for mean squared error noise detection.
    """

    def __init__(self, config: MSEDetectorConfig):
        """
        Initialize the MeanErrorNoiseDetectionHelper object.

        Parameters:
            threshold (float): The threshold for noise detection.

        Returns:
            MeanErrorNoiseDetectionHelper: A MeanErrorNoiseDetectionHelper object.
        """
        self.config = config
        self.previous_frame = None

    def register_previous_frame(self, previous_frame: np.ndarray) -> None:
        """
        Register the previous frame for mean error calculation.

        Parameters:
            previous_frame (np.ndarray): The previous frame to compare against.
        """
        self.previous_frame = previous_frame

    def find_invalid_area(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Process a single frame and verify if it is valid.

        Parameters:
            frame (np.ndarray): The frame to process.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is valid
            and the processed frame.
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            return False, np.zeros_like(frame, dtype=np.uint8)
        noisy, mask = self._detect_with_mean_error(frame)
        return noisy, mask

    def _detect_with_mean_error(self, current_frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Detect noise using mean error between current and previous frames.

        Returns:
            Tuple[bool, np.ndarray]: A boolean indicating if the frame is noisy and the noise mask.
        """
        if self.previous_frame is None:
            return False, np.zeros_like(current_frame, dtype=np.uint8)

        current_flat = current_frame.astype(np.int16).flatten()
        previous_flat = self.previous_frame.astype(np.int16).flatten()

        buffer_indices = FrameSplitter.get_buffer_shape(
            current_frame.shape[1], current_frame.shape[0], self.config.device_config.px_per_buffer
        ) + [
            current_frame.size
        ]  # Ensure final boundary is included

        noisy_mask = np.ones_like(current_flat, dtype=np.uint8)
        has_noise = False

        for start_idx, end_idx in zip(buffer_indices[:-1], buffer_indices[1:]):
            for sub_start in range(
                end_idx - self.config.buffer_split, start_idx, -self.config.buffer_split
            ):
                mean_error = np.mean(
                    np.abs(current_flat[sub_start:end_idx] - previous_flat[sub_start:end_idx])
                )

                if mean_error > self.config.threshold:
                    noisy_mask[sub_start:end_idx] = 0
                    has_noise = True
                    break

        return has_noise, noisy_mask.reshape(current_frame.shape)


class FrequencyMaskHelper(BaseSingleFrameHelper):
    """
    Helper class for frequency masking operations.
    """

    def __init__(self, height: int, width: int, freq_mask_config: FrequencyMaskingConfig):
        """
        Initialize the FreqMaskHelper object and generate a frequency mask.

        Parameters:
            height (int): The height of the image.
            width (int): The width of the image.
            freq_mask_config (FrequencyMaskingConfig): Configuration for frequency masking

        Returns:
            FreqMaskHelper: A FreqMaskHelper object.
        """
        self._height = height
        self._width = width
        self._freq_mask_config = freq_mask_config
        self._freq_mask = self._gen_freq_mask()

    @property
    def freq_mask(self) -> np.ndarray:
        """
        Get the frequency mask.

        Returns:
            np.ndarray: The frequency mask.
        """
        return self._freq_mask

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Perform FFT/IFFT to remove horizontal stripes from a single frame.

        Parameters:
            img (np.ndarray): The image to process.
            cast_f32 (bool): Cast the image to float32 before processing.

        Returns:
            np.ndarray: The filtered image

        .. todo:: Confirm if the option for casting to float32 is necessary. See issue #104.
        """
        if self._freq_mask_config.cast_float32:
            img = img.astype(np.float32)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        # Apply mask and inverse FFT
        fshift *= self.freq_mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)

        # Clip value to valid range and convert back to uint8.
        # Some cases get rounded up higher than 255.
        img_back = np.clip(np.abs(img_back), 0, np.iinfo(np.uint8).max)

        return np.uint8(img_back)

    def freq_domain(self, img: np.ndarray) -> np.ndarray:
        """
        Compute the frequency spectrum of an image.

        Parameters:
            img (np.ndarray): The image to process.

        Returns:
            np.ndarray: The frequency spectrum of the image.
        """
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)

        # Normalize the magnitude spectrum for visualization
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
        )

        return np.uint8(magnitude_spectrum)

    def _gen_freq_mask(
        self,
    ) -> np.ndarray:
        """
        Generate a mask to filter out horizontal and vertical frequencies.
        A central circular region can be removed to allow low frequencies to pass.
        """
        crow, ccol = self._height // 2, self._width // 2

        # Create an initial mask filled with ones (pass all frequencies)
        mask = np.ones((self._height, self._width), np.uint8)

        # Zero out a vertical stripe at the frequency center
        mask[
            :,
            ccol
            - self._freq_mask_config.vertical_BEF_cutoff : ccol
            + self._freq_mask_config.vertical_BEF_cutoff,
        ] = 0

        # Zero out a horizontal stripe at the frequency center
        mask[
            crow
            - self._freq_mask_config.horizontal_BEF_cutoff : crow
            + self._freq_mask_config.horizontal_BEF_cutoff,
            :,
        ] = 0

        # Define spacial low pass filter
        y, x = np.ogrid[: self._height, : self._width]
        center_mask = (x - ccol) ** 2 + (
            y - crow
        ) ** 2 <= self._freq_mask_config.spatial_LPF_cutoff_radius**2

        # Restore the center circular area to allow low frequencies to pass
        mask[center_mask] = 1

        return mask


class FrameSplitter:
    """
    Helper class for splitting frames into buffers.
    Currently only for getting the buffer shape from pixel count.
    """

    def get_buffer_shape(frame_width: int, frame_height: int, px_per_buffer: int) -> list[int]:
        """
        Get the shape of each buffer in a frame.

        Parameters:
            frame_width (int): The width of the frame.
            frame_height (int): The height of the frame.
            px_per_buffer (int): The number of pixels per buffer.

        Returns:
            list[int]: The shape of each buffer in the frame.
        """
        buffer_shape = []

        pixel_index = 0
        while pixel_index < frame_width * frame_height:
            buffer_shape.append(int(pixel_index))
            pixel_index += px_per_buffer
        logger.debug(f"Split shape: {buffer_shape}")
        return buffer_shape

"""
This module contains functions for pre-processing video data.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mio import init_logger
from mio.io import VideoReader, VideoWriter
from mio.models.frames import NamedFrame, NamedVideo
from mio.models.process import (
    DenoiseConfig,
    FreqencyMaskingConfig,
    MinimumProjectionConfig,
    NoisePatchConfig,
)
from mio.plots.video import VideoPlotter
from mio.process.frame_helper import FrequencyMaskHelper, InvalidFrameDetector
from mio.process.zstack_helper import ZStackHelper

logger = init_logger("video")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class BaseVideoProcessor:
    """
    Base class for defining an abstract video processor.

    Attributes:
    name (str): The name of the video processor.
    output_frames (list): A list of output frames.
    named_frame (NamedFrame): A NamedFrame object.
    """

    def __init__(self, name: str, output_dir: Path):
        """
        Initialize the BaseVideoProcessor object.

        Parameters:
        name (str): The name of the video processor.
        width (int): The width of the video frame.
        height (int): The height of the video frame.
        output_dir (Path): The output directory.

        Returns:
        BaseVideoProcessor: A BaseVideoProcessor object.
        """
        self.name: str = name
        self.output_dir: Path = output_dir
        self.output_video: list[np.ndarray] = []
        self.output_enable: bool = True

    @property
    def output_named_video(self) -> NamedVideo:
        """
        Get the output NamedFrame object.

        Returns:
        NamedVideo: The output NamedVideo object.
        """
        return NamedVideo(name=self.name, video=self.output_video)

    def append_output_frame(self, input_frame: np.ndarray) -> None:
        """
        Append a frame to the output_frames list.

        Parameters:
        frame (np.ndarray): The frame to append.
        """
        if input_frame is None:
            logger.warning("Attempted to append None frame, skipping.")
            return
        self.output_video.append(input_frame)

    def export_output_video(self) -> None:
        """
        Export the video to a file.
        """
        if self.output_enable:
            logger.debug(f"Exporting {self.name} video to {self.output_dir}")
            self.output_named_video.export(
                output_path=self.output_dir / "output",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} output disabled.")

    def process_frame(self) -> None:
        """
        Process a single frame. This method should be implemented in the subclass.

        Parameters:
        frame (np.ndarray): The frame to process.
        """
        raise NotImplementedError("process_frame method must be implemented in the subclass.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. This method should be overridden in the subclass.
        """
        raise NotImplementedError("batch_export_videos method must be implemented in the subclass.")


class NoisePatchProcessor(BaseVideoProcessor):
    """
    A class to apply noise patching to a video.
    """

    def __init__(
        self,
        name: str,
        noise_patch_config: NoisePatchConfig,
        output_dir: Path,
    ) -> None:
        """
        Initialize the NoisePatchProcessor object.

        Parameters:
        name (str): The name of the video processor.
        noise_patch_config (NoisePatchConfig): The noise patch configuration.
        """
        super().__init__(name, output_dir)
        self.noise_patch_config: NoisePatchConfig = noise_patch_config
        self.noise_detect_helper = InvalidFrameDetector(noise_patch_config=noise_patch_config)
        self.noise_patchs: list[np.ndarray] = []
        self.noisy_frames: list[np.ndarray] = []
        self.diff_frames: list[np.ndarray] = []
        self.dropped_frame_indices: list[int] = []

        self.output_enable: bool = noise_patch_config.output_result

        if "mean_error" in noise_patch_config.method:
            logger.warning(
                "The mean_error method is unstable and not fully tested yet." " Use with caution."
            )

    def process_frame(
        self, input_frame: np.ndarray, original_frame_index: int
    ) -> Optional[np.ndarray]:
        """
        Process a single frame.

        Parameters:
        input_frame (np.ndarray): The frame to process.
        original_frame_index (int): The original frame index from the video reader.

        Returns:
        Optional[np.ndarray]: The processed frame. If the frame is noisy, a None is returned.
        """
        if input_frame is None:
            return None

        if self.noise_patch_config.enable:
            invalid, noisy_area = self.noise_detect_helper.find_invalid_area(input_frame)

            if not invalid:
                self.append_output_frame(input_frame)
                return input_frame
            else:
                msg = f"Dropping frame {original_frame_index} of original video due to noise."
                tqdm.write(msg)
                logger.debug(msg)
                logger.debug(f"Adding noise patch for frame {original_frame_index}.")
                self.noise_patchs.append((noisy_area * np.iinfo(np.uint8).max).astype(np.uint8))
                self.noisy_frames.append(input_frame)
                self.dropped_frame_indices.append(original_frame_index)
            return None

        self.append_output_frame(input_frame)
        return input_frame

    @property
    def noise_patch_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the noise patch.
        """
        return NamedVideo(name="patched_area", video=self.noise_patchs)

    @property
    def diff_frames_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the difference frames.
        """
        if not hasattr(self.noise_patch_config, "diff_multiply"):
            diff_multiply = 1
        return NamedVideo(name=f"diff_{diff_multiply}x", video=self.diff_frames)

    @property
    def noisy_frames_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the noisy frames.
        """
        return NamedVideo(name="noisy_frames", video=self.noisy_frames)

    def export_noise_patch(self) -> None:
        """
        Export the noise patch to a file.
        """
        if not self.noise_patchs:
            logger.info(f"No noise patches to export for {self.name}.")
            return

        if self.noise_patch_config.output_noise_patch:
            logger.debug(f"Exporting {self.name} noise patch to {self.output_dir}")
            self.noise_patch_named_video.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} noise patch output disabled.")

    def export_diff_frames(self) -> None:
        """
        Export the difference frames to a file.
        """
        if self.noise_patch_config.output_diff:
            logger.info(f"Exporting {self.name} difference frames to {self.output_dir}")
            self.diff_frames_named_video.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} difference frames output disabled.")

    def export_noisy_video(self) -> None:
        """
        Export the noisy frames to a file.
        """
        if self.noise_patch_config.output_noisy_frames:
            logger.debug(f"Exporting {self.name} noisy frames to {self.output_dir}")
            self.noisy_frames_named_video.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
            # Can be anything. Just for now.
            with open(self.output_dir / f"{self.name}_dropped_frames.txt", "w") as f:
                for index in self.dropped_frame_indices:
                    f.write(f"{index}\n")
        else:
            logger.info(f"{self.name} noisy frames output disabled.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_noise_patch()
        self.export_diff_frames()
        self.export_noisy_video()


class FreqencyMaskProcessor(BaseVideoProcessor):
    """
    A class to apply frequency masking to a video.
    """

    def __init__(
        self,
        name: str,
        freq_mask_config: FreqencyMaskingConfig,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        """
        Initialize the FreqencyMaskProcessor object.

        Parameters:
        name (str): The name of the video processor.
        freq_mask_config (FreqencyMaskingConfig): The frequency masking configuration.
        """
        super().__init__(name, output_dir)
        self.freq_mask_config: FreqencyMaskingConfig = freq_mask_config
        self.freq_mask_helper = FrequencyMaskHelper(
            height=height, width=width, freq_mask_config=freq_mask_config
        )
        self.freq_domain_frames = []
        self.frame_width: int = width
        self.frame_height: int = height
        self.output_enable: bool = freq_mask_config.output_result

    @property
    def freq_mask(self) -> np.ndarray:
        """
        Get the frequency mask.
        """
        return self.freq_mask_helper.freq_mask

    @property
    def freq_mask_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the frequency mask.
        """
        return NamedFrame(name="freq_mask", frame=self.freq_mask * np.iinfo(np.uint8).max)

    @property
    def freq_domain_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the frequency domain.
        """
        return NamedVideo(name="freq_domain", video=self.freq_domain_frames)

    def process_frame(self, input_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame.

        Parameters:
        frame (np.ndarray): The frame to process.

        Returns:
        Optional[np.ndarray]: The processed frame. If the input is none, a None is returned.
        """
        if input_frame is None:
            return None
        if self.freq_mask_config.enable:
            freq_filtered_frame = self.freq_mask_helper.process_frame(img=input_frame)
            frame_freq_domain = self.freq_mask_helper.freq_domain(img=input_frame)
            self.append_output_frame(freq_filtered_frame)
            self.freq_domain_frames.append(frame_freq_domain)

            return freq_filtered_frame
        else:
            return input_frame

    def export_freq_domain_frames(self) -> None:
        """
        Export the frequency domain to a file.
        """
        if self.freq_mask_config.output_freq_domain:
            logger.debug(f"Exporting {self.name} frequency domain to {self.output_dir}")
            self.freq_domain_named_video.export(
                output_path=self.output_dir / f"{self.name}",
                fps=20,
                suffix=True,
            )
        else:
            logger.info(f"{self.name} frequency domain output disabled.")

    def export_freq_mask(self) -> None:
        """
        Export the frequency mask to a file.
        """
        if self.freq_mask_config.output_mask:
            logger.debug(f"Exporting {self.name} frequency mask to {self.output_dir}")
            self.freq_mask_named_frame.export(
                output_path=self.output_dir / f"{self.name}",
                suffix=True,
            )
        else:
            logger.info(f"{self.name} frequency mask output disabled.")

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_freq_mask()
        self.export_freq_domain_frames()


class PassThroughProcessor(BaseVideoProcessor):
    """
    A class to pass through a video.
    """

    def __init__(self, name: str, output_dir: Path):
        """
        Initialize the PassThroughProcessor object.

        Parameters:
        name (str): The name of the video processor.
        output_dir (Path): The output directory.

        Returns:
        PassThroughProcessor: A PassThroughProcessor object.
        """
        super().__init__(name, output_dir)

    @property
    def pass_through_named_video(self) -> NamedVideo:
        """
        Get the NamedFrame object for the pass through.
        """
        return NamedVideo(name=self.name, video=self.output_video)

    def process_frame(self, input_frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Parameters:
        frame (np.ndarray): The frame to process.

        Returns:
        np.ndarray: The processed frame.
        """
        if input_frame is None:
            return None
        self.append_output_frame(input_frame)
        return input_frame

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()


class MinProjSubtractProcessor(BaseVideoProcessor):
    """
    A class to apply minimum projection to a video.
    """

    def __init__(
        self,
        name: str,
        minimum_projection_config: MinimumProjectionConfig,
        output_dir: Path,
        video_frames: list[np.ndarray],
    ):
        """
        Initialize the MinimumProjectionProcessor object.

        Parameters:
        name (str): The name of the video processor.
        output_dir (Path): The output directory.

        Returns:
        MinimumProjectionProcessor: A MinimumProjectionProcessor object.
        """
        super().__init__(name, output_dir)

        if not video_frames:
            logger.warning("No frames provided for minimum projection. Skipping processing.")
            self.minimum_projection = None
            self.output_frames = []
        else:
            self.minimum_projection: np.ndarray = ZStackHelper.get_minimum_projection(video_frames)
            self.output_frames: list[np.ndarray] = [
                (frame - self.minimum_projection) for frame in video_frames
            ]

        self.minimum_projection_config: MinimumProjectionConfig = minimum_projection_config
        self.output_enable: bool = minimum_projection_config.output_result

    @property
    def min_proj_named_frame(self) -> NamedFrame:
        """
        Get the NamedFrame object for the minimum projection.
        """
        return NamedFrame(name="min_proj", frame=self.output_frames[0])

    def normalize_stack(self) -> None:
        """
        Normalize the stack of images.
        """
        if not self.output_frames:
            logger.warning(
                "No frames available in output_frames for normalization. Skipping normalization."
            )
            return

        self.output_frames = ZStackHelper.normalize_video_stack(self.output_frames)

    def export_minimum_projection(self) -> None:
        """
        Export the minimum projection to a file.
        """

    def batch_export_videos(self) -> None:
        """
        Batch export the videos to a file. Whether to export or not is controlled in each method.
        """
        self.export_output_video()
        self.export_minimum_projection()


def denoise_run(
    video_path: str,
    config: DenoiseConfig,
    csv_validation_result: Optional[tuple[bool, Optional[pd.DataFrame]]] = None,
) -> None:
    """
    Preprocess a video file and display the results.

    Parameters:
    video_path (str): The path to the video file.
    config (DenoiseConfig): The denoise configuration.
    csv_validation_result (Optional[tuple[bool, Optional[pd.DataFrame]]]):
        Result from CSV validation. If provided and valid, uses the
        pre-loaded DataFrame.
    """
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is not a required dependency of miniscope-io, to use it, "
            "install it manually or install miniscope-io with `pip install miniscope-io[plot]`"
        )

    reader = VideoReader(video_path)
    pathstem = Path(video_path).stem

    output_dir = Path.cwd() / config.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    raw_frame_processor = PassThroughProcessor(
        name=pathstem + "_raw",
        output_dir=output_dir,
    )

    output_frame_processor = PassThroughProcessor(
        name=pathstem + "_output",
        output_dir=output_dir,
    )

    noise_patch_processor = NoisePatchProcessor(
        output_dir=output_dir,
        name=pathstem + "_patch",
        noise_patch_config=config.noise_patch,
    )

    freq_mask_processor = FreqencyMaskProcessor(
        output_dir=output_dir,
        name=pathstem + "_freq_mask",
        freq_mask_config=config.frequency_masking,
        width=reader.width,
        height=reader.height,
    )

    if config.interactive_display.display_freq_mask:
        freq_mask_processor.freq_mask_named_frame.display()

    # Simple progress bar with total frame count
    total_frames = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        frame_iter = tqdm(reader.read_frames(), total=total_frames, desc="Processing frames")
        for index, frame in frame_iter:
            # Apply config end_frame if specified
            if config.end_frame and index > config.end_frame and config.end_frame != -1:
                break

            # Convert to grayscale if needed
            raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            input_frame = raw_frame_processor.process_frame(raw_frame)
            patched_frame = noise_patch_processor.process_frame(
                input_frame, original_frame_index=index
            )
            freq_masked_frame = freq_mask_processor.process_frame(patched_frame)
            _ = output_frame_processor.process_frame(freq_masked_frame)

    finally:
        reader.release()

        output_frames = output_frame_processor.output_video

        if not isinstance(output_frames, list):
            raise ValueError("Output frames must be a list.")
        for frame in output_frames:
            if not isinstance(frame, np.ndarray):
                logger.warning(f"Frame is not a numpy array: {type(frame)}")
        minimum_projection_processor = MinProjSubtractProcessor(
            name=pathstem + "min_proj",
            output_dir=output_dir,
            video_frames=output_frames,
            minimum_projection_config=config.minimum_projection,
        )
        minimum_projection_processor.normalize_stack()

        noise_patch_processor.batch_export_videos()
        freq_mask_processor.batch_export_videos()
        minimum_projection_processor.batch_export_videos()

        # Log excluded frames
        dropped_frames = noise_patch_processor.dropped_frame_indices
        if dropped_frames:
            logger.info(
                f"Excluded {len(dropped_frames)} frames due to noise: "
                f"{dropped_frames[:20]}{'...' if len(dropped_frames) > 20 else ''}"
            )
        else:
            logger.info("No frames were excluded during processing.")

        # Always modify CSV metadata to match the output video
        # Determine output video path: output_dir / "output_<name>.avi"
        output_video_name = f"output_{noise_patch_processor.name}"
        output_video_path = output_dir / f"{output_video_name}.avi"
        
        # Verify the actual output video frame count matches expected
        actual_output_frame_count = len(output_frame_processor.output_video)
        dropped_count = len(noise_patch_processor.dropped_frame_indices)
        logger.debug(
            f"Output video will have {actual_output_frame_count} frames "
            f"(input had {total_frames}, dropped {dropped_count})"
        )
        
        modified_csv_df = _modify_csv_metadata(
            video_path,
            output_video_path,
            noise_patch_processor.dropped_frame_indices,
            csv_validation_result,
        )
        
        # Validate frame count alignment after metadata modification
        if modified_csv_df is not None:
            max_metadata_index = modified_csv_df["reconstructed_frame_index"].max()
            expected_max_index = actual_output_frame_count - 1
            if max_metadata_index != expected_max_index:
                logger.warning(
                    f"Frame index mismatch: metadata max index is {max_metadata_index}, "
                    f"but output video has {actual_output_frame_count} frames "
                    f"(expected max index {expected_max_index}). "
                    f"This may indicate an off-by-one error in frame indexing."
                )

        # Export frame-timestamp metadata CSV
        if modified_csv_df is not None:
            _export_frame_timestamp_csv(output_video_path, modified_csv_df)

        if len(noise_patch_processor.output_named_video.video) == 0:
            logger.warning("No output video available for display.")
        elif (
            len(noise_patch_processor.output_named_video.video)
            < config.interactive_display.end_frame
        ):
            logger.warning(
                f"Output video has {len(noise_patch_processor.output_named_video.video)} frames."
                f" End frame for interactive plot is {config.interactive_display.end_frame}."
                " End frame for interactive plot exceeds the number of frames in the video."
                " Skipping interactive display."
            )
        elif config.interactive_display.show_videos:
            videos = [
                noise_patch_processor.output_named_video,
                freq_mask_processor.output_named_video,
                freq_mask_processor.freq_domain_named_video,
                minimum_projection_processor.min_proj_named_frame,
            ]
            video_plotter = VideoPlotter(
                videos=videos,
                start_frame=config.interactive_display.start_frame,
                end_frame=config.interactive_display.end_frame,
            )
            video_plotter.show()


def _modify_csv_metadata(
    input_video_path: str,
    output_video_path: Path,
    dropped_frame_indices: list[int],
    csv_validation_result: Optional[tuple[bool, Optional[pd.DataFrame]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Modify CSV metadata to match the denoised video by removing rows for dropped frames
    and adjusting reconstructed_frame_index.

    Parameters:
    input_video_path (str): Path to the input video file.
    output_video_path (Path): Path to the output video file.
    dropped_frame_indices (list[int]): List of frame indices that were dropped.
    csv_validation_result (Optional[tuple[bool, Optional[pd.DataFrame]]]):
        Result from CSV validation. If provided and valid, uses the
        pre-loaded DataFrame.

    Returns:
    Optional[pd.DataFrame]: The modified DataFrame, or None if CSV processing
        was skipped.
    """
    input_video_path_obj = Path(input_video_path)
    input_csv_path = input_video_path_obj.with_suffix(".csv")
    output_csv_path = output_video_path.with_suffix(".csv")

    # Use pre-validated CSV if available, otherwise read it
    if csv_validation_result is not None:
        is_valid, df = csv_validation_result
        if not is_valid or df is None:
            logger.warning(
                "CSV validation failed or CSV not available. Skipping CSV metadata modification."
            )
            return None
    else:
        # Fallback: read CSV if validation wasn't done
        if not input_csv_path.exists():
            logger.warning(
                f"CSV file not found at {input_csv_path}. Skipping CSV metadata modification."
            )
            return None

        try:
            df = pd.read_csv(input_csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file {input_csv_path}: {e}")
            return None

        if "reconstructed_frame_index" not in df.columns:
            logger.warning(
                f"CSV file {input_csv_path} does not have 'reconstructed_frame_index' column. "
                "Skipping CSV metadata modification."
            )
            return None

    # Remove dropped frames
    if not dropped_frame_indices:
        logger.info(
            "Modifying CSV metadata at %s from %s (no frames dropped, copying as-is).",
            output_csv_path,
            input_csv_path,
        )
        df_filtered = df.copy()
    else:
        logger.info(
            f"Modifying CSV metadata at {output_csv_path} "
            f"from {input_csv_path} (removing {len(dropped_frame_indices)} dropped frames)."
        )

        dropped_set = set(dropped_frame_indices)
        df_filtered = df[~df["reconstructed_frame_index"].isin(dropped_set)].copy()
        logger.info(f"Removed {len(df) - len(df_filtered)} buffers from CSV.")

        # Renumber frame indices to be continuous after removing dropped frames
        def adjust_frame_index(frame_idx: int) -> int:
            num_dropped_before = sum(1 for dropped_idx in dropped_set if dropped_idx < frame_idx)
            return frame_idx - num_dropped_before

        df_filtered["reconstructed_frame_index"] = df_filtered["reconstructed_frame_index"].apply(
            adjust_frame_index
        )

    try:
        df_filtered.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully modified CSV metadata at {output_csv_path}.")
        return df_filtered
    except Exception as e:
        logger.error(f"Failed to write output CSV file {output_csv_path}: {e}")
        raise


def _export_frame_timestamp_csv(output_video_path: Path, csv_df: pd.DataFrame) -> None:
    """
    Export a frame-timestamp CSV file mapping reconstructed_frame_index to unix timestamps.

    The CSV includes both the first and last buffer timestamps for each frame.

    Parameters:
    output_video_path (Path): Path to the output video file.
    csv_df (pd.DataFrame): The modified CSV DataFrame with
        reconstructed_frame_index and buffer_recv_unix_time.
    """
    # Check if required columns exist
    if "reconstructed_frame_index" not in csv_df.columns:
        logger.warning(
            "CSV DataFrame does not have 'reconstructed_frame_index' column. "
            "Skipping frame-timestamp CSV export."
        )
        return

    if "buffer_recv_unix_time" not in csv_df.columns:
        logger.warning(
            "CSV DataFrame does not have 'buffer_recv_unix_time' column. "
            "Skipping frame-timestamp CSV export."
        )
        return

    # Group by reconstructed_frame_index and get both first and last buffer timestamps
    frame_timestamps = (
        csv_df.groupby("reconstructed_frame_index")["buffer_recv_unix_time"]
        .agg(["min", "max"])
        .reset_index()
    )

    # Rename columns for clarity
    frame_timestamps.columns = ["frame", "timestamp_first", "timestamp_last"]

    # Sort by frame index
    frame_timestamps = frame_timestamps.sort_values("frame")

    # Create output path: same name as video but with _metadata.csv suffix
    output_csv_path = output_video_path.with_name(output_video_path.stem + "_metadata.csv")

    try:
        frame_timestamps.to_csv(output_csv_path, index=False)
        logger.info(
            "Successfully exported frame-timestamp CSV at %s (%d frames).",
            output_csv_path,
            len(frame_timestamps),
        )
    except Exception as e:
        logger.error(f"Failed to write frame-timestamp CSV file {output_csv_path}: {e}")
        raise


def crop_run(
    video_path: str,
    output_path: Optional[str] = None,
    csv_validation_result: Optional[tuple[bool, Optional[pd.DataFrame]]] = None,
    trim_start: Optional[int] = None,
    trim_end: Optional[int] = None,
) -> None:
    """
    Crop a video file by trimming frames.

    Parameters:
    video_path (str): The path to the input video file.
    output_path (Optional[str]): The path to the output video file.
        If None, defaults to input path with "_cropped" suffix.
    csv_validation_result (Optional[tuple[bool, Optional[pd.DataFrame]]]):
        Result from CSV validation. If provided and valid, uses the
        pre-loaded DataFrame.
    trim_start (Optional[int]): Start frame index for trimming (0-based,
        inclusive). If None, starts from frame 0.
    trim_end (Optional[int]): End frame index for trimming (0-based,
        inclusive). If None, ends at the last frame.
    """
    reader = VideoReader(video_path)
    input_path = Path(video_path)

    # Determine output path
    if output_path is None:
        output_path_obj = input_path.parent / f"{input_path.stem}_cropped{input_path.suffix}"
    else:
        output_path_obj = Path(output_path)

    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Get video properties
    total_frames = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = reader.cap.get(cv2.CAP_PROP_FPS)

    # Determine trim range
    trim_start_val = trim_start if trim_start is not None else 0
    trim_end_val = trim_end if trim_end is not None else total_frames - 1

    # Validate trim range
    if trim_start_val < 0:
        raise ValueError(f"trim_start must be >= 0, got {trim_start_val}")
    if trim_end_val >= total_frames:
        raise ValueError(
            f"trim_end must be < total_frames ({total_frames}), got {trim_end_val}"
        )
    if trim_start_val > trim_end_val:
        raise ValueError(
            f"trim_start ({trim_start_val}) must be <= trim_end ({trim_end_val})"
        )

    expected_output_frames = trim_end_val - trim_start_val + 1
    logger.info(
        f"Cropping video: frames {trim_start_val}-{trim_end_val} "
        f"(inclusive, {expected_output_frames} frames)"
    )

    # Create video writer
    writer = VideoWriter(path=output_path_obj, fps=fps)

    frames_written = 0
    try:
        frame_iter = tqdm(
            reader.read_frames(),
            total=total_frames,
            desc="Cropping frames"
        )
        for index, frame in frame_iter:
            # Apply trim range
            if index < trim_start_val:
                continue
            if index > trim_end_val:
                break

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Write frame
            writer.write_frame(frame)
            frames_written += 1

    finally:
        reader.release()
        writer.close()

    logger.info(
        f"Successfully cropped video: wrote {frames_written} frames to {output_path_obj}"
    )

    # Validate frame count
    if frames_written != expected_output_frames:
        logger.warning(
            f"Frame count mismatch: expected {expected_output_frames} frames, "
            f"but wrote {frames_written} frames"
        )

    # Modify CSV metadata
    _crop_csv_metadata(
        video_path,
        output_path_obj,
        csv_validation_result,
        trim_start=trim_start_val,
        trim_end=trim_end_val,
    )


def _crop_csv_metadata(
    input_video_path: str,
    output_video_path: Path,
    csv_validation_result: Optional[tuple[bool, Optional[pd.DataFrame]]] = None,
    trim_start: int = 0,
    trim_end: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Crop CSV metadata to match the cropped video by trimming and adjusting
    reconstructed_frame_index.

    Parameters:
    input_video_path (str): Path to the input video file.
    output_video_path (Path): Path to the output video file.
    csv_validation_result (Optional[tuple[bool, Optional[pd.DataFrame]]]):
        Result from CSV validation. If provided and valid, uses the
        pre-loaded DataFrame.
    trim_start (int): Start frame index for trimming (0-based, inclusive).
    trim_end (Optional[int]): End frame index for trimming (0-based,
        inclusive). If None, uses the max index from the CSV.

    Returns:
    Optional[pd.DataFrame]: The modified DataFrame, or None if CSV processing
        was skipped.
    """
    input_video_path_obj = Path(input_video_path)
    input_csv_path = input_video_path_obj.with_suffix(".csv")
    output_csv_path = output_video_path.with_suffix(".csv")

    # Use pre-validated CSV if available, otherwise read it
    if csv_validation_result is not None:
        is_valid, df = csv_validation_result
        if not is_valid or df is None:
            logger.warning(
                "CSV validation failed or CSV not available. "
                "Skipping CSV metadata modification."
            )
            return None
    else:
        # Fallback: read CSV if validation wasn't done
        if not input_csv_path.exists():
            logger.warning(
                f"CSV file not found at {input_csv_path}. "
                "Skipping CSV metadata modification."
            )
            return None

        try:
            df = pd.read_csv(input_csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file {input_csv_path}: {e}")
            return None

        if "reconstructed_frame_index" not in df.columns:
            logger.warning(
                f"CSV file {input_csv_path} does not have "
                "'reconstructed_frame_index' column. "
                "Skipping CSV metadata modification."
            )
            return None

    # Determine trim end value
    trim_end_val = (
        df["reconstructed_frame_index"].max()
        if trim_end is None
        else trim_end
    )

    # Filter to trim range and renumber starting from 0
    df_filtered = df[
        (df["reconstructed_frame_index"] >= trim_start)
        & (df["reconstructed_frame_index"] <= trim_end_val)
    ].copy()
    df_filtered["reconstructed_frame_index"] = (
        df_filtered["reconstructed_frame_index"] - trim_start
    )

    logger.info(
        f"Trimmed CSV to frames {trim_start}-{trim_end_val} "
        f"and renumbered to start from 0 "
        f"({len(df_filtered)} rows)."
    )

    try:
        df_filtered.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully modified CSV metadata at {output_csv_path}.")
        return df_filtered
    except Exception as e:
        logger.error(f"Failed to write output CSV file {output_csv_path}: {e}")
        raise

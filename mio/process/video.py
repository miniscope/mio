"""
This module contains functions for pre-processing video data.
"""

from pathlib import Path

import cv2
import numpy as np

from mio import init_logger
from mio.io import VideoReader
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
        self.output_video.append(input_frame)

    def export_output_video(self) -> None:
        """
        Export the video to a file.
        """
        if self.output_enable:
            logger.info(f"Exporting {self.name} video to {self.output_dir}")
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

    def process_frame(self, input_frame: np.ndarray) -> np.ndarray | None:
        """
        Process a single frame.

        Parameters:
        raw_frame (np.ndarray): The raw frame to process.

        Returns:
        Optional[np.ndarray]: The processed frame. If the frame is noisy, a None is returned.
        """
        if input_frame is None:
            return None

        if self.noise_patch_config.enable:
            invalid, noisy_area = self.noise_detect_helper.find_invalid_area(input_frame)

            # Handle noisy frames
            if not invalid:
                self.append_output_frame(input_frame)
                return input_frame
            else:
                index = len(self.output_video) + len(self.noise_patchs)
                logger.info(f"Dropping frame {index} of original video due to noise.")
                logger.debug(f"Adding noise patch for frame {index}.")
                self.noise_patchs.append((noisy_area * np.iinfo(np.uint8).max).astype(np.uint8))
                self.noisy_frames.append(input_frame)
                self.dropped_frame_indices.append(index)
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
            logger.info(f"Exporting {self.name} noise patch to {self.output_dir}")
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
            logger.info(f"Exporting {self.name} noisy frames to {self.output_dir}")
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

    def process_frame(self, input_frame: np.ndarray) -> np.ndarray | None:
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
            logger.info(f"Exporting {self.name} frequency domain to {self.output_dir}")
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
            logger.info(f"Exporting {self.name} frequency mask to {self.output_dir}")
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
) -> None:
    """
    Preprocess a video file and display the results.

    Parameters:
    video_path (str): The path to the video file.
    config (DenoiseConfig): The denoise configuration.
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

    try:
        for index, frame in reader.read_frames():
            if config.end_frame and index > config.end_frame and config.end_frame != -1:
                break

            raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_frame = raw_frame_processor.process_frame(raw_frame)
            patched_frame = noise_patch_processor.process_frame(input_frame)
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

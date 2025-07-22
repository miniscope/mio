"""
Pydantic models for storing frames and videos.
"""

from pathlib import Path

import cv2
import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel, Field, field_validator

from mio.io import VideoWriter
from mio.logging import init_logger

logger = init_logger("model.frames")


class NamedBaseFrame(BaseModel):
    """
    Pydantic model to store an an image or a video together with a name.
    """

    name: str = Field(
        ...,
        description="Name of the video.",
    )

    def export(self, output_path: Path | str, fps: int, suffix: bool) -> None:
        """
        Export the frame data to a file.
        The implementation needs to be defined in the derived classes.
        """
        raise NotImplementedError("Method not implemented.")


class NamedFrame(NamedBaseFrame):
    """
    Pydantic model to store an image or a video together with a name.
    """

    frame: Optional[NDArray] = Field(
        None,
        description="Frame data, if provided.",
    )

    def export(self, output_path: Path | str, suffix: bool = False) -> None:
        """
        Export the frame data to a file.
        The file name will be a concatenation of the output path and the name of the frame.
        """
        output_path = Path(output_path)
        if self.frame is None:
            logger.warning(f"No frame data provided for {self.name}. Skipping export.")
            return
        if suffix:
            output_path = output_path.with_name(output_path.stem + f"_{self.name}")
        cv2.imwrite(str(output_path.with_suffix(".png")), self.frame)
        logger.info(
            f"Writing frame to {output_path}.png: {self.frame.shape[1]}x{self.frame.shape[0]}"
        )

    def display(self, binary: bool = False) -> None:
        """
        Display the frame data in a opencv window. Press ESC to close the window.

        Parameters
        ----------
        binary : bool
            If True, the frame will be scaled to the full range of uint8.
        """
        if self.frame is None:
            logger.warning(f"No frame data provided for {self.name}. Skipping display.")
            return

        frame_to_display = self.frame
        if binary:
            frame_to_display = cv2.normalize(
                self.frame, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
            ).astype(np.uint8)
        cv2.imshow(self.name, frame_to_display)
        while True:
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Extra waitKey to properly close the window

    @field_validator("frame")
    def validate_frame_is_2d(cls, v: NDArray) -> NDArray:
        """
        Validate that the frame is a 2D array.
        """
        if v is not None and len(v.shape) != 2:
            raise ValueError("Frame must be a 2D array")
        return v


class NamedVideo(NamedBaseFrame):
    """
    Pydantic model to store a video together with a name.
    """

    video: Optional[List[NDArray]] = Field(
        None,
        description="List of frames.",
    )

    def export(self, output_path: Path | str, suffix: bool = False, fps: float = 20) -> None:
        """
        Export the frame data to a file.
        """
        if self.video is None or self.video == []:
            logger.warning(f"No frame data provided for {self.name}. Skipping export.")
            return
        output_path = Path(output_path)
        if suffix:
            output_path = output_path.with_name(output_path.stem + f"_{self.name}")
        if not all(isinstance(frame, np.ndarray) for frame in self.video):
            raise ValueError("Not all frames are numpy arrays.")
        writer = VideoWriter(
            path=output_path.with_suffix(".avi"),
            fps=fps,
        )
        logger.info(
            f"Writing video to {output_path}.avi:"
            f"{self.video[0].shape[1]}x{self.video[0].shape[0]}"
        )
        try:
            for frame in self.video:
                picture = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                writer.write_frame(picture)
        finally:
            writer.close()

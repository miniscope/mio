"""
Pydantic models for storing frames and videos.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias
from typing import Annotated as A

import cv2
import numpy as np
from numpydantic import NDArraySchema
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


FrameType: TypeAlias = A[np.ndarray, NDArraySchema(("*", "*"))]


class BaseFrame(BaseModel):
    """
    Pydantic model to store an image
    """

    frame: FrameType

    @abstractmethod
    def export(self, output_path: Path | str, suffix: bool = False) -> None:
        """
        Export the frame data to a file.
        """
        raise NotImplementedError("Method not implemented.")


class BaseVideo(BaseModel):
    """
    Pydantic model to store a video.
    """

    video: list[FrameType] = Field(
        ...,
        description="List of frames.",
    )

    @abstractmethod
    def export(self, output_path: Path | str, suffix: bool = False) -> None:
        """
        Export the frame data to a file.
        """
        raise NotImplementedError("Method not implemented.")


class NamedFrame(BaseFrame):
    """
    Pydantic model to store an image or a video together with a name.
    """

    name: str = Field(
        ...,
        description="Name of the frame.",
    )

    def export(self, output_path: Path | str, suffix: bool = False) -> None:
        """
        Export the frame data to a file.
        The file name will be a concatenation of the output path and the name of the frame.
        """
        output_path = Path(output_path)

        if suffix:
            output_path = output_path.with_name(output_path.stem + f"_{self.name}")
        cv2.imwrite(str(output_path.with_suffix(".png")), self.frame)

    def display(self, binary: bool = False) -> None:
        """
        Display the frame data in a opencv window. Press ESC to close the window.

        Parameters
        ----------
        binary : bool
            If True, the frame will be scaled to the full range of uint8.
        """
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


class NamedVideo(BaseVideo):
    """
    Pydantic model to store a video together with a name.
    """

    name: str = Field(
        ...,
        description="Name of the video.",
    )

    def export(
        self, output_path: Path | str, suffix: bool = False, fps: int = 20, force: bool = False
    ) -> None:
        """
        Export the frame data to a file.
        """
        from mio.io import VideoWriter

        output_path = Path(output_path)
        if suffix:
            output_path = output_path.with_name(output_path.stem + f"_{self.name}")
        if not all(isinstance(frame, np.ndarray) for frame in self.video):
            raise ValueError("Not all frames are numpy arrays.")
        writer = VideoWriter(path=output_path.with_suffix(".avi"), fps=fps, force=force)
        try:
            for frame in self.video:
                picture = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                writer.write_frame(picture)
        finally:
            writer.close()

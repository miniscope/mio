"""
Pydantic models for storing frames and videos.
"""

from abc import abstractmethod
from pathlib import Path
from typing import Annotated as A
from typing import Literal, TypeAlias, overload

import cv2
import numpy as np
import pandas as pd
from numpydantic import NDArray, NDArraySchema
from pydantic import BaseModel, Field, field_validator

from mio.logging import init_logger
from mio.models.sdcard import SDBufferHeader

logger = init_logger("model.frames")

FrameType: TypeAlias = A[np.ndarray, NDArraySchema(("*", "*"))]


class BaseFrame(BaseModel):
    """
    Pydantic model to store an image
    """

    frame: FrameType | None = Field(
        None,
        description="Frame data, if provided.",
    )

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

        if self.video is None or self.video == []:
            logger.warning(f"No frame data provided for {self.name}. Skipping export.")
            return
        output_path = Path(output_path)
        if suffix:
            output_path = output_path.with_name(output_path.stem + f"_{self.name}")
        if not all(isinstance(frame, np.ndarray) for frame in self.video):
            raise ValueError("Not all frames are numpy arrays.")
        writer = VideoWriter(path=output_path.with_suffix(".avi"), fps=fps, force=force)
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


class SDCardFrame(BaseModel):
    """
    An individual frame from a miniscope recording

    Typically returned from :meth:`.SDCard.read`
    """

    frame: NDArray
    headers: list[SDBufferHeader]

    @field_validator("headers")
    @classmethod
    def frame_nums_must_be_equal(cls, v: list[SDBufferHeader]) -> list[SDBufferHeader] | None:
        """
        Each frame_number field in each header must be the same
        (they come from the same frame!)
        """

        if v is not None and not all([header.frame_num != v[0].frame_num for header in v]):
            raise ValueError(f"All frame numbers should be equal! Got f{[h.frame_num for h in v]}")
        return v

    @property
    def frame_num(self) -> int | None:
        """
        Frame number for this set of headers, if headers are present
        """
        return self.headers[0].frame_num


class SDCardVideo(BaseModel):
    """
    A collection of frames from a miniscope recording
    """

    frames: list[SDCardFrame]

    @overload
    def flatten_headers(self, as_dict: Literal[False]) -> list[SDBufferHeader]: ...

    @overload
    def flatten_headers(self, as_dict: Literal[True]) -> list[dict]: ...

    def flatten_headers(self, as_dict: bool = False) -> list[dict] | list[SDBufferHeader]:
        """
        Return flat list of headers, not grouped by frame

        Args:
            as_dict (bool): If `True`, return a list of dictionaries, if `False`
                (default), return a list of :class:`.SDBufferHeader` s.
        """
        h: list[dict] | list[SDBufferHeader] = []
        for frame in self.frames:
            headers: list[dict] | list[SDBufferHeader]
            if as_dict:
                headers = [header.model_dump() for header in frame.headers]
            else:
                headers = frame.headers
            h.extend(headers)
        return h

    def to_df(self, what: Literal["headers"] = "headers") -> pd.DataFrame:
        """
        Convert frames to pandas dataframe

        Arguments:
            what ('headers'): What information from the frame to include in the df,
                currently only 'headers' is possible
        """

        if what == "headers":
            return pd.DataFrame(self.flatten_headers(as_dict=True))
        else:
            raise ValueError("Return mode not implemented!")

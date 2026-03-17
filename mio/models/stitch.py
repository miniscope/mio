"""
Models for stitching multiple recordings together.
"""

from typing import List

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from mio.logging import init_logger

logger = init_logger(name="models.stitch")


class FrameInfo(BaseModel):
    """
    Container containing information about a single frame.
    This container is oriented around the reconstructed frame index.
    """

    reconstructed_frame_index: int
    frame_num: int

    @classmethod
    def from_metadata(cls, frame_num: int, metadata: pd.DataFrame) -> "FrameInfo":
        """Create a FrameInfo instance from frame_num and metadata."""
        frame_metadata = metadata[metadata["frame_num"] == frame_num]

        if frame_metadata.empty:
            raise ValueError(f"No metadata found for frame_num {frame_num}")

        rfi_values = frame_metadata["reconstructed_frame_index"]
        if rfi_values.nunique() == 1:
            reconstructed_frame_index = rfi_values.iloc[0]
        else:
            reconstructed_frame_index = rfi_values.mode()[0]
            msg = (
                f"Reconstructed frame index is not the same "
                f"for all buffers in frame {frame_num}. "
                f"Using the majority reconstructed_frame_index: {reconstructed_frame_index}"
            )
            tqdm.write(msg)
            logger.debug(msg)

        return cls(
            frame_num=frame_num,
            reconstructed_frame_index=reconstructed_frame_index,
        )


class DebugRecord(BaseModel):
    """
    Row schema for debug metadata emitted during stitching.

    The field order defines the CSV header order.
    """

    debug_frame_index: int
    stitched_frame_index: int
    frame_num: int
    selected_video: str
    compare_video: str
    selected_num_buffers: int
    selected_black_padding: int
    compare_num_buffers: int
    compare_black_padding: int
    diff_pixels: int
    selected_edge_score: float
    compare_edge_score: float
    metadata_tie: bool
    selection_mode: str = "metadata"
    selected_is_noisy: bool | None = None
    compare_is_noisy: bool | None = None

    @classmethod
    def header(cls) -> List[str]:
        """Return CSV header preserving declared field order."""
        return list(cls.model_fields.keys())

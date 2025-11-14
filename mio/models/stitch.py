"""
Models for stitching multiple recordings together.
"""

from typing import List, Optional

import pandas as pd
from numpydantic import NDArray
from pydantic import BaseModel

from mio.logging import init_logger

logger = init_logger(name="models.stitch")


class BufferInfo(BaseModel):
    """
    Container containing information about a single buffer.
    This container is oriented around the buffer receive index.
    """

    buffer_recv_index: int
    buffer_count: int
    frame_buffer_count: int
    timestamp: int
    pixel_count: int
    black_padding_px: int
    buffer_recv_unix_time: float
    pixel_data: Optional[NDArray] = None


class FrameInfo(BaseModel):
    """
    Container containing information about a single frame.
    This container is oriented around the reconstructed frame index.
    """

    reconstructed_frame_index: int
    frame_num: int

    buffer_info_list: List[BufferInfo] = []

    @classmethod
    def from_metadata(cls, frame_num: int, metadata: pd.DataFrame) -> "FrameInfo":
        """Create a FrameInfo instance from frame_num and metadata."""
        # Find all buffer entries for this frame_num
        frame_metadata = metadata[metadata["frame_num"] == frame_num]

        if frame_metadata.empty:
            raise ValueError(f"No metadata found for frame_num {frame_num}")

        if all(
            frame_metadata["reconstructed_frame_index"].iloc[0] == x
            for x in frame_metadata["reconstructed_frame_index"]
        ):
            reconstructed_frame_index = frame_metadata["reconstructed_frame_index"].iloc[0]
        else:
            # Get the majority reconstructed_frame_index
            reconstructed_frame_index = frame_metadata["reconstructed_frame_index"].mode()[0]
            logger.warning(
                f"Reconstructed frame index is not the same "
                f"for all buffers in frame {frame_num}. "
                f"Using the majority reconstructed_frame_index: {reconstructed_frame_index}"
            )

        buffer_info_list = []

        # Iterate through all buffer entries for this frame
        # sort based on buffer_recv_index
        frame_metadata = frame_metadata.sort_values(by="buffer_recv_index")
        for i in range(len(frame_metadata)):
            buffer_info_list.append(
                BufferInfo(
                    buffer_recv_index=frame_metadata["buffer_recv_index"].iloc[i],
                    buffer_count=frame_metadata["buffer_count"].iloc[i],
                    frame_buffer_count=frame_metadata["frame_buffer_count"].iloc[i],
                    timestamp=frame_metadata["timestamp"].iloc[i],
                    pixel_count=frame_metadata["pixel_count"].iloc[i],
                    black_padding_px=frame_metadata["black_padding_px"].iloc[i],
                    buffer_recv_unix_time=frame_metadata["buffer_recv_unix_time"].iloc[i],
                )
            )

        return cls(
            frame_num=frame_num,
            reconstructed_frame_index=reconstructed_frame_index,
            buffer_info_list=buffer_info_list,
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

    @classmethod
    def header(cls) -> List[str]:
        """Return CSV header preserving declared field order."""
        return list(cls.model_fields.keys())

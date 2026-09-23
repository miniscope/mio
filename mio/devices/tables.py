"""Shared metadata models"""

from __future__ import annotations

import pandera.pandas as pa

from mio.models import Container, Table


class TimestampTable(pa.DataFrameModel):
    """Summary of timestamps per frame for a recording"""

    frame: int
    timestamp_first: float
    timestamp_last: float


class NoiseTable(pa.DataFrameModel):
    """
    Scores for noise values for a recording,
    produced by :func:`mio.process.video.score_noise`
    """

    reconstructed_frame_index: int = pa.Field(ge=0, coerce=True)
    black_area: int | None = pa.Field(ge=0, coerce=True)
    gradient: float | None = pa.Field(ge=0, coerce=True)


class StitchRecord(Container):
    """
    Row schema for debug metadata emitted during stitching.

    The field order defines the CSV header order.
    """

    index: int
    frame_num: int | None = None
    selected_video: str
    compare_video: str | None = None
    selected_num_buffers: int
    selected_black_padding: int
    selected_black_pixels: int
    selected_noisy_pixels: int
    compare_num_buffers: int | None = None
    compare_black_padding: int | None = None
    compare_black_pixels: int | None = None
    compare_noisy_pixels: int | None = None
    selected_edge_score: float | None = None
    compare_edge_score: float | None = None

    @classmethod
    def header(cls) -> list[str]:
        """Return CSV header preserving declared field order."""
        return list(cls.model_fields.keys())


class StitchTable(Table):
    """Table model for stitching scores"""

    _RECORD_MODEL = StitchRecord

    index: int = pa.Field(ge=0, coerce=True)
    frame_num: int = pa.Field(default=None, ge=0, coerce=True, nullable=True)
    selected_video: str
    compare_video: str = pa.Field(default=None, coerce=True, nullable=True)
    selected_num_buffers: int
    selected_black_padding: int
    selected_black_pixels: int
    selected_noisy_pixels: int
    compare_num_buffers: int = pa.Field(default=None, ge=0, coerce=True, nullable=True)
    compare_black_padding: int = pa.Field(default=None, ge=0, coerce=True, nullable=True)
    compare_black_pixels: int = pa.Field(default=None, ge=0, coerce=True, nullable=True)
    compare_noisy_pixels: int = pa.Field(default=None, ge=0, coerce=True, nullable=True)
    selected_edge_score: float = pa.Field(default=None, coerce=True, nullable=True)
    compare_edge_score: float = pa.Field(default=None, coerce=True, nullable=True)

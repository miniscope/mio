"""
Buffer-wise stitching of multiple data streams based on device timestamps.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
This is still hardcoded around the StreamDevConfig metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mio.io import BufferedCSVWriter, VideoReader, VideoWriter
from mio.logging import init_logger
from mio.models.stitch import DebugRecord, FrameInfo

logger = init_logger(name="stitch")


def score_edges(frame: np.ndarray) -> float:
    """Negative of total Sobel gradient magnitude (higher is better)."""
    gx = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3)
    return -float(np.abs(gx).sum() + np.abs(gy).sum())


@dataclass
class CandidateFrame:
    """A single candidate frame from one recording for a given frame_num."""

    recording: RecordingData
    frame: np.ndarray
    num_buffers: int
    sum_black_padding: int
    metadata_rows: pd.DataFrame
    edge_score: float

    @property
    def metadata_score(self) -> tuple[int, int]:
        """Higher is better: more buffers, less black padding.
        A bit overkill but left this for future extension.
        """
        return (self.num_buffers, -self.sum_black_padding)


def select_best_candidate(candidates: list[CandidateFrame]) -> tuple[int, bool]:
    """
    Pick the best candidate using metadata scoring with edge-score tiebreak.

    Metadata score: (num_buffers, -sum_black_padding) lexicographically.
    Ties are broken by edge score (less sharp = better, i.e. less noise).
    Returns (best_index, was_tie).
    """
    top_score = max(c.metadata_score for c in candidates)
    tied = [i for i, c in enumerate(candidates) if c.metadata_score == top_score]
    best_idx = tied[0]
    is_tie = len(tied) > 1
    if is_tie:
        tied_scores = [candidates[i].edge_score for i in tied]
        best_idx = tied[int(np.argmax(tied_scores))]
    return best_idx, is_tie


class RecordingData:
    """Class for a single stream's data (video + metadata)."""

    def __init__(
        self,
        video_path: Path,
        csv_path: Path,
    ) -> None:
        self.video_path: Path = video_path
        self.csv_path: Path = csv_path
        self._video_reader: VideoReader | None = None
        self._metadata: pd.DataFrame | None = None

    @classmethod
    def from_video_paths(cls, video_paths: list[Path]) -> list[RecordingData]:
        """Build a list of RecordingData from video paths, inferring companion CSVs."""
        recordings: list[RecordingData] = []
        for video_path in video_paths:
            csv_path = video_path.with_suffix(".csv")
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found for {video_path}: {csv_path}")
            recordings.append(cls(video_path=video_path, csv_path=csv_path))
        return recordings

    @property
    def video_reader(self) -> VideoReader:
        """Get or create the video reader."""
        if self._video_reader is None:
            self._video_reader = VideoReader(str(self.video_path))
        return self._video_reader

    @property
    def metadata(self) -> pd.DataFrame:
        """Get or load metadata CSV as DataFrame."""
        if self._metadata is None:
            self._metadata = pd.read_csv(self.csv_path)
        return self._metadata


class RecordingDataBundle:
    """Class for a bundle of recording data."""

    def __init__(
        self,
        recordings: list[RecordingData],
        stitched_video_writer: VideoWriter,
        debug_video_writer: VideoWriter | None = None,
        combined_csv_path: Path | None = None,
        debug_csv_path: Path | None = None,
    ) -> None:
        self.recordings: list[RecordingData] = recordings
        self.stitched_video_writer: VideoWriter = stitched_video_writer
        self.debug_video_writer: VideoWriter | None = debug_video_writer
        self.combined_csv_path: Path | None = combined_csv_path
        self._metadata_parts: list[pd.DataFrame] = []
        self._combined_frame_num: list[int] | None = None
        self._out_frame_index: int = 0
        self.debug_csv_writer: BufferedCSVWriter | None = None
        self._debug_frame_index: int = 0
        if debug_csv_path is not None:
            self.debug_csv_writer = BufferedCSVWriter(
                debug_csv_path, header=DebugRecord.header(), buffer_size=100
            )

    @property
    def combined_frame_num(self) -> list[int]:
        """
        Get the combined frame_num.
        This is a list of unique frame_nums across all recordings.
        """
        if self._combined_frame_num is None:
            combined = list(
                dict.fromkeys(fn for r in self.recordings for fn in r.metadata["frame_num"])
            )
            self._combined_frame_num = combined
        return self._combined_frame_num

    def _collect_candidates(self, frame_num: int) -> list[CandidateFrame]:
        """Read frames and metadata scores for all recordings that have *frame_num*."""
        candidates: list[CandidateFrame] = []
        for recording in self.recordings:
            rows = recording.metadata[recording.metadata["frame_num"] == frame_num]
            if rows.empty:
                continue
            frame_info = FrameInfo.from_metadata(frame_num=frame_num, metadata=recording.metadata)
            frame = recording.video_reader.read_frame(frame_info.reconstructed_frame_index)
            if frame is None:
                continue
            num_buffers = int(len(rows))
            sum_black = int(rows["black_padding_px"].fillna(0).sum())
            candidates.append(
                CandidateFrame(
                    recording=recording,
                    frame=frame,
                    num_buffers=num_buffers,
                    sum_black_padding=sum_black,
                    metadata_rows=rows,
                    edge_score=score_edges(frame),
                )
            )
        return candidates

    def _write_debug(
        self,
        frame_num: int,
        candidates: list[CandidateFrame],
        selected_idx: int,
        is_tie: bool,
    ) -> int:
        """Write debug composites for frames that differ from the selected one.

        Returns the number of debug frames written.
        """
        if len(candidates) <= 1:
            return 0

        selected = candidates[selected_idx]
        others = [(i, c) for i, c in enumerate(candidates) if i != selected_idx]
        if all(np.array_equal(selected.frame, c.frame) for _, c in others):
            return 0

        writes = 0
        for idx, cand in others:
            diff_mask = (selected.frame != cand.frame).astype(np.uint8) * 255
            diff_pixels = int(np.count_nonzero(diff_mask))
            msg = (
                f"Frames are not the same for frame {frame_num} "
                f"(Rec {selected_idx} vs Rec {idx}): {diff_pixels} px differ"
            )
            tqdm.write(msg)
            logger.debug(msg)

            if self.debug_video_writer is not None:
                composite = np.vstack([selected.frame, cand.frame, diff_mask])
                self.debug_video_writer.write_frame(composite)
                writes += 1

            if self.debug_csv_writer is not None:
                record = DebugRecord(
                    debug_frame_index=self._debug_frame_index,
                    stitched_frame_index=self._out_frame_index,
                    frame_num=frame_num,
                    selected_video=selected.recording.video_path.name,
                    compare_video=cand.recording.video_path.name,
                    selected_num_buffers=selected.num_buffers,
                    selected_black_padding=selected.sum_black_padding,
                    compare_num_buffers=cand.num_buffers,
                    compare_black_padding=cand.sum_black_padding,
                    diff_pixels=diff_pixels,
                    selected_edge_score=selected.edge_score,
                    compare_edge_score=cand.edge_score,
                    metadata_tie=bool(is_tie),
                )
                self.debug_csv_writer.append(record.model_dump())
                self._debug_frame_index += 1

        return writes

    def _write_stitched(
        self,
        candidates: list[CandidateFrame],
        selected_idx: int,
    ) -> None:
        """Write the selected frame and its metadata to the stitched outputs."""
        selected = candidates[selected_idx]
        self.stitched_video_writer.write_frame(selected.frame)

        rows = selected.metadata_rows.copy()
        rows["reconstructed_frame_index"] = self._out_frame_index
        self._metadata_parts.append(rows)
        self._out_frame_index += 1

    def _finalize(self) -> None:
        """Close writers and flush combined CSV."""
        self.stitched_video_writer.close()
        if self.debug_video_writer is not None:
            self.debug_video_writer.close()
        if self.debug_csv_writer is not None:
            self.debug_csv_writer.close()
        if self.combined_csv_path is not None and self._metadata_parts:
            pd.concat(self._metadata_parts, ignore_index=True).to_csv(
                self.combined_csv_path, index=False
            )

    def stitch_recordings(self) -> None:
        """Stitch recordings by iterating unique frame_nums and selecting the best frame."""
        stitched_writes = 0
        debug_writes = 0
        frame_iter = tqdm(self.combined_frame_num, desc="Stitching frames")

        for frame_num in frame_iter:
            valid_pairs = self._collect_candidates(frame_num)
            if not valid_pairs:
                continue
            selected_idx, is_tie = select_best_candidate(valid_pairs)
            debug_writes += self._write_debug(frame_num, valid_pairs, selected_idx, is_tie)
            self._write_stitched(valid_pairs, selected_idx)
            stitched_writes += 1

        self._finalize()
        logger.info(
            f"Stitch completed: stitched_writes={stitched_writes}, debug_writes={debug_writes}"
        )

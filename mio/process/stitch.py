"""
Buffer-wise stitching and concatenation of multiple data streams.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
It also provides concatenation of sequential recording segments from the same DAQ.
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

    def _build_timestamp_matches(
        self, threshold_ms: float = 25.0
    ) -> list[dict[int, int]]:
        """
        Match frames across recordings by nearest unix timestamp.

        For each recording, compute per-frame timestamp as the max
        buffer_recv_unix_time for each reconstructed_frame_index.

        Uses recording[0] as the reference. For each frame in ref,
        find nearest frame in each other recording within threshold.

        Returns list of dicts: [{rec_idx: reconstructed_frame_index, ...}, ...]
        One entry per matched frame set, ordered by ref recording's frame order.
        """
        threshold_s = threshold_ms / 1000.0

        # Build per-frame timestamp arrays for each recording
        per_rec_timestamps: list[tuple[np.ndarray, np.ndarray]] = []
        for rec in self.recordings:
            df = rec.metadata
            grouped = df.groupby("reconstructed_frame_index")["buffer_recv_unix_time"].max()
            frame_indices = grouped.index.values
            timestamps = grouped.values
            sort_order = np.argsort(timestamps)
            per_rec_timestamps.append((frame_indices[sort_order], timestamps[sort_order]))

        ref_indices, ref_timestamps = per_rec_timestamps[0]
        matches: list[dict[int, int]] = []

        for i, (ref_idx, ref_ts) in enumerate(zip(ref_indices, ref_timestamps)):
            match: dict[int, int] = {0: int(ref_idx)}
            for rec_num in range(1, len(self.recordings)):
                other_indices, other_timestamps = per_rec_timestamps[rec_num]
                pos = np.searchsorted(other_timestamps, ref_ts)

                best_dist = float("inf")
                best_idx = -1
                for candidate_pos in [pos - 1, pos]:
                    if 0 <= candidate_pos < len(other_timestamps):
                        dist = abs(other_timestamps[candidate_pos] - ref_ts)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = int(other_indices[candidate_pos])

                if best_dist <= threshold_s:
                    match[rec_num] = best_idx

            if len(match) > 1:
                matches.append(match)

        return matches

    def _collect_candidates_by_index(
        self, frame_indices: dict[int, int]
    ) -> list[CandidateFrame]:
        """Collect candidates using reconstructed_frame_index directly."""
        candidates: list[CandidateFrame] = []
        for rec_num, rfi in frame_indices.items():
            recording = self.recordings[rec_num]
            rows = recording.metadata[recording.metadata["reconstructed_frame_index"] == rfi]
            if rows.empty:
                continue
            frame = recording.video_reader.read_frame(rfi)
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

    def stitch_recordings(
        self,
        matching_method: str = "frame_num",
        timestamp_threshold_ms: float = 25.0,
    ) -> None:
        """Stitch recordings by selecting the best frame per matched position.

        Parameters
        ----------
        matching_method : str
            ``"frame_num"`` (default) matches by device frame_num.
            ``"timestamp"`` matches by nearest ``buffer_recv_unix_time``.
        timestamp_threshold_ms : float
            Max time difference in ms for timestamp matching (default 25).
        """
        stitched_writes = 0
        debug_writes = 0

        if matching_method == "timestamp":
            matches = self._build_timestamp_matches(
                threshold_ms=timestamp_threshold_ms
            )
            frame_iter = tqdm(matches, desc="Stitching frames (timestamp)")
            for match in frame_iter:
                candidates = self._collect_candidates_by_index(match)
                if not candidates:
                    continue
                selected_idx, is_tie = select_best_candidate(candidates)
                # Use first recording's frame index as label for debug
                frame_label = match.get(0, 0)
                debug_writes += self._write_debug(
                    frame_label, candidates, selected_idx, is_tie
                )
                self._write_stitched(candidates, selected_idx)
                stitched_writes += 1
        else:
            frame_iter = tqdm(self.combined_frame_num, desc="Stitching frames")
            for frame_num in frame_iter:
                valid_pairs = self._collect_candidates(frame_num)
                if not valid_pairs:
                    continue
                selected_idx, is_tie = select_best_candidate(valid_pairs)
                debug_writes += self._write_debug(
                    frame_num, valid_pairs, selected_idx, is_tie
                )
                self._write_stitched(valid_pairs, selected_idx)
                stitched_writes += 1

        self._finalize()
        logger.info(
            f"Stitch completed: stitched_writes={stitched_writes}, debug_writes={debug_writes}"
        )


def concat_recordings(
    recordings: list[RecordingData],
    output_video_path: Path,
    output_csv_path: Path,
    fps: int = 20,
) -> None:
    """Concatenate sequential recording segments into a single video + CSV.

    Each recording's frames are appended in order. The CSV metadata is merged
    with ``reconstructed_frame_index`` renumbered to be contiguous across all
    segments.

    Parameters
    ----------
    recordings : list[RecordingData]
        Ordered list of recording segments to concatenate.
    output_video_path : Path
        Path for the combined output AVI.
    output_csv_path : Path
        Path for the combined output CSV.
    fps : int
        Frames per second for the output video.
    """
    video_writer = VideoWriter(path=output_video_path, fps=fps)
    metadata_parts: list[pd.DataFrame] = []
    rfi_offset = 0
    total_frames = 0

    for i, rec in enumerate(tqdm(recordings, desc="Concatenating segments")):
        # Copy all video frames
        seg_frames = 0
        for _, frame in rec.video_reader.read_frames():
            video_writer.write_frame(frame)
            seg_frames += 1

        # Offset reconstructed_frame_index in metadata
        df = rec.metadata.copy()
        max_rfi = int(df["reconstructed_frame_index"].max())
        df["reconstructed_frame_index"] = df["reconstructed_frame_index"] + rfi_offset
        metadata_parts.append(df)

        logger.info(
            f"Segment {i}: {rec.video_path.name} — "
            f"{seg_frames} frames, rfi_offset={rfi_offset}"
        )
        rfi_offset += max_rfi + 1
        total_frames += seg_frames

    video_writer.close()

    combined_df = pd.concat(metadata_parts, ignore_index=True)
    combined_df.to_csv(output_csv_path, index=False)

    logger.info(
        f"Concat completed: {total_frames} frames from "
        f"{len(recordings)} segments -> {output_video_path}"
    )

"""
Buffer-wise stitching of multiple data streams based on device timestamps.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
This is still hardcoded around the StreamDevConfig metadata fields.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mio.io import BufferedCSVWriter, VideoReader, VideoWriter
from mio.logging import init_logger
from mio.models.stitch import DebugRecord, FrameInfo

logger = init_logger(name="stitch")


def score_metadata(num_buffers: int, sum_black_padding: int) -> Tuple[int, int]:
    """Return a tuple score for metadata (higher is better lexicographically)."""
    return (num_buffers, -sum_black_padding)


def score_edges(frame: np.ndarray) -> float:
    """Negative of total Sobel gradient magnitude (higher is better)."""
    gx = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3)
    total_grad = int(np.abs(gx).sum() + np.abs(gy).sum())
    return -float(total_grad)


def most_proper_metadata(
    valid_pairs: List[Tuple["RecordingData", np.ndarray, int, int]],
) -> Tuple[int, List[int], bool]:
    """
    Select less broken frames using metadata scoring.

    Uses score_metadata(num_buffers, sum_black_padding) and returns the
    index of a best-scoring candidate, the list of tied candidate indices,
    and whether there was a tie.
    """
    if not valid_pairs:
        return 0, [], False

    scored = [
        (i, score_metadata(num_buffers=v[2], sum_black_padding=v[3]))
        for i, v in enumerate(valid_pairs)
    ]
    # Sort descending by score tuple
    scored.sort(key=lambda t: t[1], reverse=True)
    top_score = scored[0][1]
    candidates = [i for i, s in scored if s == top_score]
    best_idx = candidates[0]
    is_tie = len(candidates) > 1
    return best_idx, candidates, is_tie


def most_proper_frame(frame_list: List[np.ndarray]) -> Tuple[int, List[float]]:
    """
    Select using the edge-based scoring function score_edges(frame).
    Returns the best index and the list of scores.
    """
    if not frame_list:
        return 0, []

    # Ensure all frames are 2D arrays with identical shapes
    shapes = [f.shape for f in frame_list if isinstance(f, np.ndarray)]
    if len(shapes) != len(frame_list) or len(set(shapes)) != 1:
        return 0, [float("-inf")] * len(frame_list)

    scores = [score_edges(f) for f in frame_list]
    best_idx = int(np.argmax(scores))
    return best_idx, scores


class RecordingData:
    """Class for a single stream's data (video + metadata)."""

    def __init__(
        self,
        video_path: Path,
        csv_path: Path,
    ) -> None:
        self.video_path: Path = video_path
        self.csv_path: Path = csv_path
        self._video_reader: Optional[VideoReader] = None
        self._metadata: Optional[pd.DataFrame] = None

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
        recordings: List[RecordingData],
        combined_video_writer: VideoWriter,
        debug_video_writer: Optional[VideoWriter] = None,
        combined_csv_path: Optional[Path] = None,
        debug_csv_path: Optional[Path] = None,
    ) -> None:
        self.recordings: List[RecordingData] = recordings
        self.combined_video_writer: VideoWriter = combined_video_writer
        self.debug_video_writer: Optional[VideoWriter] = debug_video_writer
        self.combined_csv_path: Optional[Path] = combined_csv_path
        self.combined_metadata: Optional[pd.DataFrame] = None
        self._combined_frame_num: Optional[List[int]] = None
        self._out_frame_index: int = 0
        # Debug CSV writer
        self.debug_csv_writer: Optional[BufferedCSVWriter] = None
        self._debug_frame_index: int = 0
        if debug_csv_path is not None:
            self.debug_csv_writer = BufferedCSVWriter(
                debug_csv_path, header=DebugRecord.header(), buffer_size=100
            )

    @property
    def combined_frame_num(self) -> List[int]:
        """
        Get the combined frame_num.
        This is a list of unique frame_nums across all recordings.
        """
        if self._combined_frame_num is None:
            seen: set = set()
            combined: List[int] = []
            for recording in self.recordings:
                for fn in recording.metadata["frame_num"]:
                    if fn not in seen:
                        seen.add(fn)
                        combined.append(fn)
            self._combined_frame_num = combined
        return self._combined_frame_num

    def _collect_candidates(
        self, frame_num: int
    ) -> List[Tuple[RecordingData, np.ndarray, int, int]]:
        """Read frames and metadata scores for all recordings that have *frame_num*."""
        valid_pairs: List[Tuple[RecordingData, np.ndarray, int, int]] = []
        for recording in self.recordings:
            if frame_num not in recording.metadata["frame_num"].values:
                continue
            frame_info = FrameInfo.from_metadata(
                frame_num=frame_num, metadata=recording.metadata
            )
            frame = recording.video_reader.read_frame(frame_info.reconstructed_frame_index)
            if frame is None:
                continue
            rows = recording.metadata[recording.metadata["frame_num"] == frame_num]
            num_buffers = int(len(rows))
            sum_black = (
                int(rows["black_padding_px"].fillna(0).sum())
                if "black_padding_px" in rows.columns
                else 0
            )
            valid_pairs.append((recording, frame, num_buffers, sum_black))
        return valid_pairs

    @staticmethod
    def _select_best(
        valid_pairs: List[Tuple[RecordingData, np.ndarray, int, int]],
    ) -> Tuple[int, bool]:
        """Pick the best candidate index using metadata scoring + edge tiebreak."""
        best_idx, candidates, is_tie = most_proper_metadata(valid_pairs)
        if is_tie:
            candidate_frames = [valid_pairs[i][1] for i in candidates]
            rel_idx, _ = most_proper_frame(candidate_frames)
            best_idx = candidates[int(rel_idx)]
        return best_idx, is_tie

    def _write_debug(
        self,
        frame_num: int,
        valid_pairs: List[Tuple[RecordingData, np.ndarray, int, int]],
        selected_idx: int,
        is_tie: bool,
    ) -> int:
        """Write debug composites for frames that differ from the selected one.

        Returns the number of debug frames written.
        """
        frames = [vp[1] for vp in valid_pairs]
        if len(frames) <= 1:
            return 0

        base = frames[selected_idx]
        others = [(i, f) for i, f in enumerate(frames) if i != selected_idx]
        if all(np.array_equal(base, f) for _, f in others):
            return 0

        writes = 0
        for idx, frame in others:
            if base.shape != frame.shape:
                msg = (
                    f"Frames differ for frame {frame_num}"
                    f": shape {base.shape} vs {frame.shape}"
                )
                tqdm.write(msg)
                logger.debug(msg)
                continue

            diff_mask = (base != frame).astype(np.uint8) * 255
            diff_pixels = int(np.count_nonzero(diff_mask))
            msg = (
                f"Frames are not the same for frame {frame_num} "
                f"(Rec {selected_idx} vs Rec {idx}): {diff_pixels} px differ"
            )
            tqdm.write(msg)
            logger.debug(msg)

            if self.debug_video_writer is not None:
                try:
                    composite = np.vstack([base, frame, diff_mask])
                    self.debug_video_writer.write_frame(composite)
                    writes += 1
                except Exception as e:
                    msg = f"Failed to write composite for frame {frame_num}: {e}"
                    tqdm.write(msg)
                    logger.warning(msg)

            if self.debug_csv_writer is not None:
                base_rec, _, base_buffers, base_black = valid_pairs[selected_idx]
                rec_i, _, nbuff_i, nblack_i = valid_pairs[idx]
                record = DebugRecord(
                    debug_frame_index=self._debug_frame_index,
                    stitched_frame_index=self._out_frame_index,
                    frame_num=frame_num,
                    selected_video=base_rec.video_path.name,
                    compare_video=rec_i.video_path.name,
                    selected_num_buffers=base_buffers,
                    selected_black_padding=base_black,
                    compare_num_buffers=nbuff_i,
                    compare_black_padding=nblack_i,
                    diff_pixels=diff_pixels,
                    selected_edge_score=score_edges(base),
                    compare_edge_score=score_edges(frame),
                    metadata_tie=bool(is_tie),
                )
                self.debug_csv_writer.append(record.model_dump())
                self._debug_frame_index += 1

        return writes

    def _write_stitched(
        self,
        frame_num: int,
        valid_pairs: List[Tuple[RecordingData, np.ndarray, int, int]],
        selected_idx: int,
    ) -> bool:
        """Write the selected frame and its metadata to the stitched outputs.

        Returns True on success, False on failure.
        """
        selected_frame = valid_pairs[selected_idx][1]
        try:
            self.combined_video_writer.write_frame(selected_frame)
        except Exception as e:
            msg = (
                f"Failed to write stitched frame {frame_num}: {e}"
                f" (shape={getattr(selected_frame, 'shape', None)}"
                f" dtype={getattr(selected_frame, 'dtype', None)})"
            )
            tqdm.write(msg)
            logger.warning(msg)
            return False

        try:
            selected_recording = valid_pairs[selected_idx][0]
            rows = selected_recording.metadata[
                selected_recording.metadata["frame_num"] == frame_num
            ].copy()
            rows["reconstructed_frame_index"] = self._out_frame_index
            if self.combined_metadata is None:
                self.combined_metadata = rows
            else:
                self.combined_metadata = pd.concat(
                    [self.combined_metadata, rows], ignore_index=True
                )
            self._out_frame_index += 1
        except Exception as e:
            msg = f"Failed to collect metadata for frame {frame_num}: {e}"
            tqdm.write(msg)
            logger.debug(msg)
            return False

        return True

    def _finalize(self) -> None:
        """Close writers and flush combined CSV."""
        try:
            if hasattr(self.combined_video_writer, "close"):
                self.combined_video_writer.close()
            if self.debug_video_writer is not None:
                self.debug_video_writer.close()
            if self.debug_csv_writer is not None:
                self.debug_csv_writer.close()
        finally:
            if self.combined_csv_path is not None and self.combined_metadata is not None:
                self.combined_metadata.to_csv(self.combined_csv_path, index=False)

    def stitch_recordings(self) -> None:
        """Stitch recordings by iterating unique frame_nums and selecting the best frame."""
        stitched_writes = 0
        debug_writes = 0
        frame_iter = tqdm(self.combined_frame_num, desc="Stitching frames")

        for frame_num in frame_iter:
            try:
                valid_pairs = self._collect_candidates(frame_num)
                if not valid_pairs:
                    continue
                selected_idx, is_tie = self._select_best(valid_pairs)
                debug_writes += self._write_debug(
                    frame_num, valid_pairs, selected_idx, is_tie
                )
                if self._write_stitched(frame_num, valid_pairs, selected_idx):
                    stitched_writes += 1
            except Exception as e:
                msg = f"Error processing frame_num {frame_num}: {e}"
                tqdm.write(msg)
                logger.debug(msg)

        self._finalize()
        logger.info(
            f"Stitch completed: stitched_writes={stitched_writes}, debug_writes={debug_writes}"
        )

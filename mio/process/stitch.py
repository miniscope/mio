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
from mio.models.process import NoisePatchConfig
from mio.models.stitch import DebugRecord, FrameInfo
from mio.process.frame_helper import InvalidFrameDetector

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
    is_noisy: bool | None = None

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


def select_best_candidate_noise_aware(
    candidates: list[CandidateFrame],
) -> tuple[int, bool] | None:
    """
    Pick the best candidate using noise detection results.

    Returns (best_index, was_tie) or None if all candidates are noisy (skip frame).

    Logic:
    - If all candidates are noisy, return None (skip this frame)
    - If exactly one is clean, pick it
    - If multiple are clean, fall back to metadata scoring among clean ones
    """
    clean = [i for i, c in enumerate(candidates) if not c.is_noisy]
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0], False
    # Multiple clean candidates — use metadata scoring among them
    clean_candidates = [candidates[i] for i in clean]
    best_among_clean, was_tie = select_best_candidate(clean_candidates)
    return clean[best_among_clean], was_tie


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
        noise_config: NoisePatchConfig | None = None,
        debug_dir: Path | None = None,
        fps: int = 20,
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
        self._noise_detector: InvalidFrameDetector | None = None
        self._debug_dir: Path | None = debug_dir
        self._fps: int = fps
        # Noise tracking for summary
        self._per_rec_noisy: list[int] = [0] * len(recordings)
        self._both_noisy_indices: list[int] = []  # matched position indices
        self._both_noisy_writer: VideoWriter | None = None
        self._total_matched: int = 0
        if noise_config is not None:
            if "mean_error" in (noise_config.method or []):
                raise ValueError(
                    "mean_error detection is not supported during stitching "
                    "(it requires sequential frames from a single recording). "
                    "Use only 'gradient' and/or 'black_area' methods."
                )
            self._noise_detector = InvalidFrameDetector(noise_config)
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                self._both_noisy_writer = VideoWriter(
                    path=debug_dir / "both_broken.avi", fps=fps
                )
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

    def _detect_noise(self, frame: np.ndarray) -> bool | None:
        """Run noise detection on a frame if a detector is configured."""
        if self._noise_detector is None:
            return None
        is_noisy, _ = self._noise_detector.find_invalid_area(frame)
        return is_noisy

    def _collect_candidates(self, frame_num: int) -> list[CandidateFrame]:
        """Read frames and metadata scores for all recordings that have *frame_num*."""
        candidates: list[CandidateFrame] = []
        skip_edge_score = self._noise_detector is not None
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
                    edge_score=0.0 if skip_edge_score else score_edges(frame),
                    is_noisy=self._detect_noise(frame),
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
                selection_mode = "noise_aware" if self._noise_detector is not None else "metadata"
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
                    selection_mode=selection_mode,
                    selected_is_noisy=selected.is_noisy,
                    compare_is_noisy=cand.is_noisy,
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
        if self._both_noisy_writer is not None:
            self._both_noisy_writer.close()
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
        skip_edge_score = self._noise_detector is not None
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
                    edge_score=0.0 if skip_edge_score else score_edges(frame),
                    is_noisy=self._detect_noise(frame),
                )
            )
        return candidates

    def _print_noise_summary(
        self, stitched_writes: int, skipped_both_noisy: int
    ) -> None:
        """Print a terminal summary of noise statistics."""
        total = self._total_matched
        if total == 0:
            return
        fps = self._fps
        both_pct = 100.0 * skipped_both_noisy / total
        print(f"\n{'=' * 60}")
        print("STITCH NOISE SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total matched frames:    {total}")
        print(f"  Stitched (output):       {stitched_writes} "
              f"({stitched_writes / fps:.1f}s, {stitched_writes / fps / 3600:.2f}h)")
        print(f"  Both broken (skipped):   {skipped_both_noisy} "
              f"({both_pct:.2f}%, {skipped_both_noisy / fps:.1f}s)")
        for i, rec in enumerate(self.recordings):
            noisy = self._per_rec_noisy[i]
            pct = 100.0 * noisy / total if total > 0 else 0
            print(f"  Rec {i} noisy ({rec.video_path.name}): "
                  f"{noisy} ({pct:.2f}%)")
        print(f"{'=' * 60}\n")

    def _generate_noise_report(
        self, stitched_writes: int, skipped_both_noisy: int
    ) -> None:
        """Generate a drop analysis PNG in the debug directory."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping noise report plot")
            return

        if not self._both_noisy_indices and skipped_both_noisy == 0:
            return

        total = self._total_matched
        fps = self._fps
        dropped_indices = np.array(self._both_noisy_indices)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Timeline
        ax1 = axes[0]
        time_hours = dropped_indices / fps / 3600.0
        total_hours = total / fps / 3600.0
        ax1.vlines(time_hours, 0, 1, colors="red", alpha=0.5, linewidth=0.5)
        ax1.set_xlim(0, total_hours)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        ax1.set_xlabel("Time (hours)")
        ax1.set_title(
            f"Both-broken frames timeline "
            f"({len(dropped_indices)} dropped across {total} total)"
        )

        # Plot 2: Run length distribution
        ax2 = axes[1]
        if len(dropped_indices) > 0:
            diffs = np.diff(dropped_indices)
            runs = []
            current_run = 1
            for d in diffs:
                if d == 1:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            max_run = max(runs) if runs else 1
            bins = range(1, min(max_run + 2, 102))
            ax2.hist(runs, bins=bins, color="steelblue", edgecolor="black",
                     linewidth=0.5)
            ax2.axvline(x=fps, color="red", linestyle="--", alpha=0.7,
                        label=f"1 second ({fps} frames)")
            ax2.legend()
        ax2.set_xlabel("Consecutive dropped frames (run length)")
        ax2.set_ylabel("Number of events")
        ax2.set_title(
            f"Distribution of drop run lengths "
            f"({len(dropped_indices)} events)"
        )

        # Plot 3: Drop density (1-minute bins)
        ax3 = axes[2]
        time_minutes = dropped_indices / fps / 60.0
        total_minutes = total / fps / 60.0
        if total_minutes > 0:
            bins_minutes = np.arange(0, total_minutes + 1, 1)
            ax3.hist(time_minutes, bins=bins_minutes, color="orangered",
                     edgecolor="none", rwidth=0.8)
        ax3.set_xlabel("Time (minutes)")
        ax3.set_ylabel("Dropped frames per minute")
        ax3.set_title("Drop density over time (1-minute bins)")

        plt.tight_layout()
        out_path = self._debug_dir / "noise_report.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        logger.info(f"Noise report saved to {out_path}")

    def stitch_recordings(
        self,
        matching_method: str = "frame_num",
        timestamp_threshold_ms: float = 25.0,
        max_frames: int = -1,
    ) -> None:
        """Stitch recordings by selecting the best frame per matched position.

        Parameters
        ----------
        matching_method : str
            ``"frame_num"`` (default) matches by device frame_num.
            ``"timestamp"`` matches by nearest ``buffer_recv_unix_time``.
        timestamp_threshold_ms : float
            Max time difference in ms for timestamp matching (default 25).
        max_frames : int
            Maximum number of frames to write. -1 means all frames.
        """
        stitched_writes = 0
        debug_writes = 0
        skipped_both_noisy = 0
        use_noise_aware = self._noise_detector is not None
        match_position = 0

        def _select(candidates: list[CandidateFrame]) -> tuple[int, bool] | None:
            if use_noise_aware:
                return select_best_candidate_noise_aware(candidates)
            return select_best_candidate(candidates)

        def _track_noise(candidates: list[CandidateFrame], position: int) -> None:
            """Track per-recording noisy counts."""
            if not use_noise_aware:
                return
            for i, c in enumerate(candidates):
                if c.is_noisy and i < len(self._per_rec_noisy):
                    self._per_rec_noisy[i] += 1

        def _handle_both_noisy(candidates: list[CandidateFrame], position: int) -> None:
            """Write all candidate frames to both-broken AVI for manual review."""
            self._both_noisy_indices.append(position)
            if self._both_noisy_writer is not None:
                for c in candidates:
                    self._both_noisy_writer.write_frame(c.frame)

        if matching_method == "timestamp":
            matches = self._build_timestamp_matches(
                threshold_ms=timestamp_threshold_ms
            )
            if max_frames > 0:
                matches = matches[:max_frames]
            frame_iter = tqdm(matches, desc="Stitching frames (timestamp)")
            for match in frame_iter:
                candidates = self._collect_candidates_by_index(match)
                if not candidates:
                    match_position += 1
                    continue
                self._total_matched += 1
                _track_noise(candidates, match_position)
                result = _select(candidates)
                if result is None:
                    skipped_both_noisy += 1
                    _handle_both_noisy(candidates, match_position)
                    match_position += 1
                    continue
                selected_idx, is_tie = result
                frame_label = match.get(0, 0)
                debug_writes += self._write_debug(
                    frame_label, candidates, selected_idx, is_tie
                )
                self._write_stitched(candidates, selected_idx)
                stitched_writes += 1
                match_position += 1
        else:
            frame_nums = self.combined_frame_num
            if max_frames > 0:
                frame_nums = frame_nums[:max_frames]
            frame_iter = tqdm(frame_nums, desc="Stitching frames")
            for frame_num in frame_iter:
                valid_pairs = self._collect_candidates(frame_num)
                if not valid_pairs:
                    match_position += 1
                    continue
                self._total_matched += 1
                _track_noise(valid_pairs, match_position)
                result = _select(valid_pairs)
                if result is None:
                    skipped_both_noisy += 1
                    _handle_both_noisy(valid_pairs, match_position)
                    match_position += 1
                    continue
                selected_idx, is_tie = result
                debug_writes += self._write_debug(
                    frame_num, valid_pairs, selected_idx, is_tie
                )
                self._write_stitched(valid_pairs, selected_idx)
                stitched_writes += 1
                match_position += 1

        self._finalize()
        msg = f"Stitch completed: stitched_writes={stitched_writes}, debug_writes={debug_writes}"
        if skipped_both_noisy > 0:
            msg += f", skipped_both_noisy={skipped_both_noisy}"
        logger.info(msg)

        # Print noise summary and generate plots if noise-aware
        if use_noise_aware:
            self._print_noise_summary(stitched_writes, skipped_both_noisy)
            if self._debug_dir is not None:
                self._generate_noise_report(stitched_writes, skipped_both_noisy)


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

"""
Buffer-wise stitching and concatenation of multiple data streams.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
It also provides concatenation of sequential recording segments from the same DAQ.
This is still hardcoded around the StreamDevConfig metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from mio.io import BufferedCSVWriter, VideoWriter
from mio.logging import init_logger
from mio.models.dataset import Dataset, Recording, StitchedRecording

logger = init_logger(name="stitch")


def align(recordings: list[Recording]) -> pd.DataFrame:
    """
    Create an alignment map by frame index.

    Note that this **does not** align by timestamp!
    it assumes that there is some ``frame_num`` in the metadata col for each of the recordings
    that comes from some common device.
    """
    metadatas: dict[str, pd.DataFrame] = {r.name: r.metadata for r in recordings}
    if not all(isinstance(m, pd.DataFrame) for m in metadatas.values()):
        raise ValueError("All recordings must have metadata csvs to align them")
    if not all(
        "frame_num" in m.columns and "reconstructed_frame_index" in m.columns
        for m in metadatas.values()
    ):
        raise ValueError("All recordings must have frame_num and reconstructed_frame_index columns")

    # find the full set of frames in all the recordings
    frame_set = set()
    for m in metadatas.values():
        frame_set |= set(m["frame_num"])

    # aggregate mappings from frame nums to the reconstructed frame index
    frame_maps = {
        name: df[["frame_num", "reconstructed_frame_index"]]
        .groupby("frame_num")
        .agg(lambda m: m.mode())
        .sort_values("frame_num")
        for name, df in metadatas.items()
    }

    # outer join gets us the alignment map
    names = sorted(frame_maps.keys())
    first_name = names.pop(0)
    aligned = (
        frame_maps[first_name].copy().rename(columns={"reconstructed_frame_index": first_name})
    )
    for name in names:
        aligned = aligned.merge(frame_maps[name], on="frame_num", how="outer")
        aligned.rename(columns={"reconstructed_frame_index": name}, inplace=True)

    aligned = aligned.astype("Int64")
    # popping the index twice first makes `frame_num` into a column, then an `index` column
    aligned = aligned.reset_index().reset_index()
    return aligned


def stitch(
    recordings: list[Recording],
    dataset: Dataset | None = None,
    debug_video: bool = False,
    output_dir: Path | None = None,
    progress: bool = False,
    force: bool = False,
) -> StitchedRecording:
    """
    Combine multiple recordings from the same device into a single recording
    by selecting the best matching frame from each.

    Note that this is specialized to multiple recordings of *exactly* the same thing -
    i.e. the same underlying data stream was recorded through multiple sensors,
    as is done with streaming wireless data.
    It does not handle stitching or aligning videos that were recorded with *different* devices,
    for that use the :attr:`.Dataset.alignment_map`
    which aligns simultaneous frames in different recordings.
    """
    if len(recordings) != 2:
        raise NotImplementedError("Only stitching two videos simultaneously is supported!")

    if dataset is None:
        dataset = Dataset.from_recordings(recordings)
    output_dir = dataset.path if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ensure the alignment map exists and includes these recordings
    dataset = dataset.align(recordings)

    name = "__".join(sorted([r.name for r in recordings])) + "_stitched"
    metadata_path = output_dir / (name + ".csv")
    scores_path = output_dir / (name + "_scores.csv")
    video_path = output_dir / (name + ".avi")
    debug_video_path = output_dir / (name + "_debug.avi")
    scores_writer = BufferedCSVWriter(scores_path, header=StitchRecord.header(), force=force)
    metadata_writer = BufferedCSVWriter(
        metadata_path, header=list(recordings[0].metadata.columns), force=force
    )
    video_writer = VideoWriter(
        video_path, int(recordings[0].video.video.get(cv2.CAP_PROP_FPS)), force=force
    )
    if debug_video:
        debug_video_writer = VideoWriter(
            debug_video_path, int(recordings[0].video.video.get(cv2.CAP_PROP_FPS)), force=force
        )
    else:
        debug_video_writer = None

    if progress:
        iterator = tqdm(dataset.alignment_map.iterrows(), total=len(dataset.alignment_map))
    else:
        iterator = dataset.alignment_map.iterrows()

    try:
        recs = {rec.name: rec for rec in recordings}
        for _, row in iterator:
            candidates = []
            frames = []
            for name, rec in recs.items():
                if pd.isna(row[name]):
                    if debug_video_writer is not None:
                        frames.append(np.zeros(rec.video.shape[1:], dtype=np.uint8))
                    continue
                buffer_rows = rec.metadata[rec.metadata["reconstructed_frame_index"] == row[name]]
                frames.append(rec.video[int(row[name])])
                candidates.append(
                    CandidateFrame(
                        recording=rec,
                        frame=frames[-1],
                        num_buffers=len(buffer_rows),
                        sum_black_padding=int(buffer_rows["black_padding_px"].fillna(0).sum()),
                        metadata_rows=buffer_rows,
                    )
                )
            result = _select_best_candidate(candidates, row["index"], row["frame_num"])
            selected = [c for c in candidates if c.recording.name == result.selected_video][0]
            if debug_video_writer is not None:
                debug_frame = np.zeros_like(frames[0], dtype=np.uint8)
                debug_frame[frames[1] != frames[0]] = 255
                frames.append(debug_frame)
                debug_video_writer.write_frame(np.hstack(frames))
            video_writer.write_frame(selected.frame)
            for _, md_row in selected.metadata_rows.iterrows():
                row_dict = dict(md_row)
                row_dict["reconstructed_frame_index"] = row["index"]
                metadata_writer.append(row_dict)
            scores_writer.append(result.model_dump())

    finally:
        metadata_writer.close()
        scores_writer.close()
        video_writer.close()
        if progress:
            iterator.close()

    return StitchedRecording.from_video(video_path)


@dataclass
class CandidateFrame:
    """A single candidate frame from one recording for a given frame_num."""

    recording: Recording
    frame: np.ndarray
    num_buffers: int
    sum_black_padding: int
    metadata_rows: pd.DataFrame
    _edge_score: float | None = field(default=None, repr=False)

    @property
    def edge_score(self) -> float:
        """Lazy edge score — only computed on first access (Sobel is expensive)."""
        if self._edge_score is None:
            self._edge_score = _score_edges(self.frame)
        return self._edge_score

    @property
    def metadata_score(self) -> tuple[int, int]:
        """Higher is better: more buffers, less black padding.
        A bit overkill but left this for future extension.
        """
        return (self.num_buffers, -self.sum_black_padding)


def _score_edges(frame: np.ndarray) -> float:
    """Negative of total Sobel gradient magnitude (higher is better)."""
    gx = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3)
    return -float(np.abs(gx).sum() + np.abs(gy).sum())


class StitchRecord(BaseModel):
    """
    Row schema for debug metadata emitted during stitching.

    The field order defines the CSV header order.
    """

    index: int
    frame_num: int
    selected_video: str
    compare_video: str | None = None
    selected_num_buffers: int
    selected_black_padding: int
    compare_num_buffers: int | None = None
    compare_black_padding: int | None = None
    selected_edge_score: float | None = None
    compare_edge_score: float | None = None

    @classmethod
    def header(cls) -> list[str]:
        """Return CSV header preserving declared field order."""
        return list(cls.model_fields.keys())


def _select_best_candidate(
    candidates: list[CandidateFrame], index: int, frame_num: int
) -> StitchRecord:
    """
    Pick the best candidate using metadata scoring with edge-score tiebreak.

    Metadata score: (num_buffers, -sum_black_padding) lexicographically.
    Ties are broken by edge score (less sharp = better, i.e. less noise).
    Returns (best_index, was_tie).
    """
    kwargs = {"index": index, "frame_num": frame_num}

    if len(candidates) > 2:
        raise NotImplementedError("Can only compare two candidates at once!")
    elif len(candidates) == 1:
        return StitchRecord(
            selected_video=candidates[0].recording.name,
            selected_num_buffers=candidates[0].num_buffers,
            selected_black_padding=candidates[0].sum_black_padding,
            **kwargs,
        )

    top_score = max(c.metadata_score for c in candidates)
    tied = [i for i, c in enumerate(candidates) if c.metadata_score == top_score]
    best_idx = tied[0]
    is_tie = len(tied) > 1
    if is_tie:
        tied_scores = [candidates[i].edge_score for i in tied]
        best_idx = tied[int(np.argmax(tied_scores))]
        selected = candidates[best_idx]
        other = candidates[1 if best_idx == 0 else 0]

        kwargs["selected_edge_score"] = selected.edge_score
        kwargs["compare_edge_score"] = other.edge_score
    else:
        selected = candidates[best_idx]
        other = candidates[1 if best_idx == 0 else 0]

    return StitchRecord(
        selected_video=selected.recording.name,
        selected_num_buffers=selected.num_buffers,
        selected_black_padding=selected.sum_black_padding,
        compare_video=other.recording.name,
        compare_num_buffers=other.num_buffers,
        compare_black_padding=other.sum_black_padding,
        **kwargs,
    )


def concat_recordings(
    recordings: list[Recording],
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
        total_frames = rec.video.shape[0]
        for n in range(total_frames):
            frame = rec.video[0]
            video_writer.write_frame(frame)
            seg_frames += 1

        # Offset reconstructed_frame_index in metadata
        df = rec.metadata.copy()
        max_rfi = int(df["reconstructed_frame_index"].max())
        df["reconstructed_frame_index"] = df["reconstructed_frame_index"] + rfi_offset
        metadata_parts.append(df)

        logger.info(
            f"Segment {i}: {rec.video.path.name} — " f"{seg_frames} frames, rfi_offset={rfi_offset}"
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


def _build_timestamp_matches(recordings, threshold_ms: float = 25.0) -> list[dict[int, int]]:
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
    for rec in recordings:
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
        for rec_num in range(1, len(recordings)):
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

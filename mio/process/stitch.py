"""
Buffer-wise stitching and concatenation of multiple data streams.

This module combines multiple recordings (AVI video + metadata CSV) by selecting
the best buffers from each stream using gradient noise detection.
It also provides concatenation of sequential recording segments from the same DAQ.
This is still hardcoded around the StreamDevConfig metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm, trange

from mio.io import BufferedCSVWriter, VideoWriter
from mio.logging import init_logger
from mio.models.dataset import Dataset, Recording, StitchedRecording, paths_from_video
from mio.models.process import NoisePatchConfig

logger = init_logger(name="stitch")


def align(recordings: list[Recording]) -> pd.DataFrame:
    """
    Create an alignment map by frame index.

    Note that this **is not** a general alignment method yet -
    this is specialized to the case of stitching two recordings of the same underlying data source,
    as is done when we record multiple FPGA sensors in the miniscope zero.
    Please raise an issue if you need a general frame alignment mechanisms.

    We have two kinds of alignment, depending on the structure of the metadata:

    * If all the recordings have continuously incrementing `frame_num`s,
      we align by the ``frame_num``.
      The `frame_num` is given by the device, and is the same across recordings,
      even if they start at different times (and capture different ranges of frame nums).
      This is an **outer join**, keeping all frames
    * If the recordings have *discontinuous* ``frame_num`` s,
      e.g. if the device was restarted during acquisition, we align by the acquisition timestamp.
      This assumes that the system times are closely matching
      (specifically, more closely than the interval between successive frames in the recording).
      This is an **inner join**, where we only keep frames where we can align timestamps.
    """
    if not all(isinstance(r.metadata, pd.DataFrame) for r in recordings):
        raise ValueError("All recordings must have metadata csvs to align them")
    if not all(
        "frame_num" in r.metadata.columns and "reconstructed_frame_index" in r.metadata.columns
        for r in recordings
    ):
        raise ValueError("All recordings must have frame_num and reconstructed_frame_index columns")

    if not any(_has_discontinuous_runs(r.metadata["frame_num"]) for r in recordings):
        logger.debug("Using frame-num based alignment")
        return _align_by_frame(recordings)
    else:
        logger.debug("Using time-based alignment")
        return _align_by_time(recordings)


def _align_by_frame(recordings: list[Recording]) -> pd.DataFrame:
    """Align metadata by the frame_num column"""
    metadatas: dict[str, pd.DataFrame] = {r.name: r.metadata for r in recordings}
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


def _align_by_time(recordings: list[Recording]) -> pd.DataFrame:
    """
    Align by the nearest unix timestamp.

    Use the mean of the timestamps from the buffers to get frames that have the most overlap.

    This could be made an outer join by just keeping the leading and trailing rows,
    and filtering rows with NaNs in the interior regions of buffer_recv_unix_time_x and y
    but leaving as inner for now to match existing timestamp match fn.

    the inner join functions like "when both frames mutually pick each other as their closest frame"
    which filters blippy frames that are very short.

    I **think** but have not tested that doing this triple merge method is faster
    than nested iteration, esp for longer recordings, since these are all vector ops.
    """
    metadatas: dict[str, pd.DataFrame] = {r.name: r.metadata for r in recordings}
    time_maps = {
        name: df.groupby("reconstructed_frame_index")["buffer_recv_unix_time"].mean().reset_index()
        for name, df in metadatas.items()
    }

    # inner join on closest mean timestamp value
    names = sorted(time_maps.keys())
    last_name = names.pop(0)
    aligned = time_maps[last_name].copy().rename(columns={"reconstructed_frame_index": last_name})
    for name in names:
        # merge left and right, then take the inner match
        left = pd.merge_asof(
            aligned, time_maps[name], on="buffer_recv_unix_time", direction="nearest"
        )
        right = pd.merge_asof(
            time_maps[name], aligned, on="buffer_recv_unix_time", direction="nearest"
        )
        left.rename(columns={"reconstructed_frame_index": name}, inplace=True)
        right.rename(columns={"reconstructed_frame_index": name}, inplace=True)

        # merge on the frame indexes from the left and right -
        # align when both sides agree they are the closest,
        # dropping extras from glitches/sampling rate differences
        aligned = pd.merge(left, right, "inner", on=[last_name, name])

        # keep the left's times, keeping them anchored rather than wandering in each recording
        aligned = aligned[[c for c in aligned.columns if c != "buffer_recv_unix_time_y"]]
        aligned.rename(columns={"buffer_recv_unix_time_x": "buffer_recv_unix_time"}, inplace=True)
        last_name = name

    aligned = aligned.astype({k: "Int64" for k in metadatas})
    # popping the index gives us the 'index' column
    aligned = aligned.reset_index()
    return aligned


def stitch(
    recordings: list[Recording],
    noise_config: NoisePatchConfig | None = None,
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

    Args:
        recordings (list[Recording]): List of recordings to stitch.
        noise_config (NoisePatchConfig | None): Configuration used for scoring
            noise per frame (with :func:`.score_noise` ). If None, use defaults
        dataset (Dataset | None): existing dataset, e.g. with existing alignment mapping
        output_dir: (Path | None): where to write stitched video and metadata,
            if None, same as recording directory
        progress (bool): Show a progress bar. Default ``False``
        force (bool): Overwrite existing stitched video and metadata CSV files
    """
    if len(recordings) != 2:
        raise NotImplementedError("Only stitching two videos simultaneously is supported!")

    # ensure that the recordings have noise scores
    # (does not recompute if they already exist)
    for rec in recordings:
        rec.score_noise(config=noise_config, progress=progress, force=force)

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
                noise_row = rec.noise[rec.noise["reconstructed_frame_index"] == row[name]].iloc[0]
                black_pixels = int(noise_row["black_area"]) if "black_area" in noise_row else 0
                noisy_pixels = int(noise_row["noisy_area"]) if "noisy_area" in noise_row else 0

                frames.append(rec.video[int(row[name])])
                candidates.append(
                    CandidateFrame(
                        recording=rec,
                        frame=frames[-1],
                        num_buffers=len(buffer_rows),
                        sum_black_padding=int(buffer_rows["black_padding_px"].fillna(0).sum()),
                        black_pixels=black_pixels,
                        noisy_pixels=noisy_pixels,
                        metadata_rows=buffer_rows,
                    )
                )
            result = _select_best_candidate(candidates, row["index"], row.get("frame_num"))
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
    black_pixels: int
    noisy_pixels: int
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
        # To discuss - we are probably double counting padding and missing buffers,
        # but keeping similar to existing method until we can decide what we want here -jls
        return (self.num_buffers, -self.sum_black_padding - self.black_pixels - self.noisy_pixels)


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


def _select_best_candidate(
    candidates: list[CandidateFrame], index: int, frame_num: int | None = None
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
            selected_black_pixels=candidates[0].black_pixels,
            selected_noisy_pixels=candidates[0].noisy_pixels,
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
        selected_black_pixels=selected.black_pixels,
        selected_noisy_pixels=selected.noisy_pixels,
        compare_video=other.recording.name,
        compare_num_buffers=other.num_buffers,
        compare_black_padding=other.sum_black_padding,
        compare_black_pixels=other.black_pixels,
        compare_noisy_pixels=other.noisy_pixels,
        **kwargs,
    )


def concat_recordings(
    recordings: list[Recording], output_video_path: Path, progress: bool = False
) -> Recording:
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
    progress : bool
        Show a progress bar
    """
    fps = int(recordings[0].video.video.get(cv2.CAP_PROP_FPS))
    video_writer = VideoWriter(path=output_video_path, fps=fps)
    metadata_parts: list[pd.DataFrame] = []
    rfi_offset = 0
    total_frames = 0

    recs = (
        tqdm(enumerate(recordings), desc="Concatenating recordings", position=0)
        if progress
        else enumerate(recordings)
    )
    frame_iter_cls = partial(trange, position=1) if progress else range
    try:
        for i, rec in recs:
            # Copy all video frames
            seg_frames = 0
            total_frames = rec.video.shape[0]

            for n in frame_iter_cls(total_frames):
                frame = rec.video[n]
                video_writer.write_frame(frame)
                seg_frames += 1

            # Offset reconstructed_frame_index in metadata
            df = rec.metadata.copy()
            max_rfi = int(df["reconstructed_frame_index"].max())
            df["reconstructed_frame_index"] = df["reconstructed_frame_index"] + rfi_offset
            metadata_parts.append(df)

            logger.debug(
                "Segment %s: %s — %s frames, rfi_offset=%s",
                i,
                rec.video.path.name,
                seg_frames,
                rfi_offset,
            )
            rfi_offset += max_rfi + 1
            total_frames += seg_frames
    finally:
        video_writer.close()
        if progress:
            recs.close()

    combined_df = pd.concat(metadata_parts, ignore_index=True)
    combined_df.to_csv(paths_from_video(output_video_path)["metadata"], index=False)

    logger.debug(
        "Concat completed: %s frames from %s segments -> %s",
        total_frames,
        len(recordings),
        output_video_path,
    )
    return Recording.from_video(output_video_path)


def _has_discontinuous_runs(series: pd.Series) -> bool:
    """
    Check if a metadata series has multiple discontinuous series of values:
    e.g. when acquiring frames and the counter is reset.

    Ignores single-row discontinuities like e.g. from a single buffer having an incorrect frame_num
    """
    # we need the initial NaN for alignment below, so don't drop it yet -
    # filtering NaNs is presumably cheaper than diffing
    diff = series.diff().fillna(0)
    # fast "no" if the whole series is continuous
    if (diff <= 1).all() and (diff >= 0).all():
        return False

    # filter to ignore singleton blips
    # e.g. frame_num breaks in one buffer,
    # find numbers that don't return to the prior number or number + 1 in the subsequent rows
    blips = np.logical_and(
        ~diff.between(0, 1),
        np.logical_or(diff == diff.shift(-1) * -1, diff == (diff.shift(-1) - 1) * -1),
    )

    # now check if there are any longer lasting discontinuities
    diff = series[~blips].diff().dropna()
    return bool((~diff.between(0, 1)).any())

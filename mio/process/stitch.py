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

from mio.io import BufferedCSVWriter, VideoReader, VideoWriter
from mio.logging import init_logger
from mio.models.stitch import DebugRecord, FrameInfo
from mio.models.stream import StreamDevConfig
from mio.process.stitch_helper import make_combined_list
from mio.stream_daq import StreamDaq

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

    scores: List[float] = []
    for f in frame_list:
        if not isinstance(f, np.ndarray) or f.ndim != 2:
            scores.append(float("-inf"))
        else:
            scores.append(score_edges(f))

    best_idx = int(np.argmax(scores))
    return best_idx, scores


class RecordingData:
    """Class for a single stream's data (video + metadata)."""

    def __init__(
        self,
        video_path: Path,
        csv_path: Path,
        device_config: Optional[StreamDevConfig] = None,
    ) -> None:
        self.video_path: Path = video_path
        self.csv_path: Path = csv_path
        self._device_config: Optional[StreamDevConfig] = device_config
        self._daq: Optional[StreamDaq] = None
        self._buffer_npix: Optional[List[int]] = None
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

    @property
    def daq(self) -> StreamDaq:
        """
        Get the stream daq.

        .. todo::
            Re-think this, though it is probablynot critical.
            We just need the buffer_npix list to reconstruct the buffers from the frame.
            It could make sense to just make buffer_npix static method on StreamDaq.
        """
        if self._daq is None:
            self._daq = StreamDaq(self._device_config)
        return self._daq

    @property
    def buffer_npix(self) -> List[int]:
        """
        Get the buffer npix.
        """
        if self.daq is None:
            raise ValueError("StreamDaq is not initialized")
        if self._buffer_npix is None:
            self._buffer_npix = self.daq.buffer_npix
        return self._buffer_npix

    def get_frame_index_from_timestamp(self, timestamp: int) -> int:
        """
        Get the frame index from the timestamp
        """
        if timestamp not in self.metadata["timestamp"].values:
            raise ValueError(f"Timestamp {timestamp} not found in metadata")
        return self.metadata[self.metadata["timestamp"] == timestamp]["frame_index"].iloc[0]

    def get_frame_from_timestamp(self, timestamp: int) -> np.ndarray:
        """
        Get the frame from the timestamp
        """
        frame_index = self.get_frame_index_from_timestamp(timestamp)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        _, frame = self.video_cap.read()
        return frame

    def get_buffer_metadata_from_frame_index(self, frame_index: int) -> List[int]:
        """
        Get the timestamps and buffer frame indices from the frame index.
        """
        timestamp_list = []
        buffer_frame_index_list = []
        for i in range(len(self.buffer_npix)):
            timestamp_list.append(
                self.metadata[self.metadata["frame_index"] == frame_index]["timestamp"].iloc[i]
            )
            buffer_frame_index_list.append(
                self.metadata[self.metadata["frame_index"] == frame_index][
                    "buffer_frame_index"
                ].iloc[i]
            )
        return timestamp_list, buffer_frame_index_list

    def get_frame_as_buffer_time_array(self, timestamp: int) -> FrameInfo:
        """
        Get the frame as a list of buffers and a list of timestamps.

        The buffers are reconstructed from the frame using the buffer_npix list.

        .. todo::
            Handle missing buffers.
        """
        frame_index = self.get_frame_index_from_timestamp(timestamp)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.video_cap.read()
        buffer_list = []
        timestamp_list = []
        buffer_frame_index_list = []
        timestamp_list, buffer_frame_index_list = self.get_buffer_metadata_from_frame_index(
            frame_index
        )
        for i in range(len(self.buffer_npix)):
            buffer_list.append(frame[i * self.buffer_npix[i] : (i + 1) * self.buffer_npix[i]])
        return FrameInfo(
            buffer_list=buffer_list,
            timestamp_list=timestamp_list,
            buffer_frame_index_list=buffer_frame_index_list,
        )

    def get_frame_info_from_frame_num(self, frame_num: int) -> FrameInfo:
        """
        Get the frame info from the frame num.
        """
        # Use the FrameInfo class method which handles the metadata parsing
        return FrameInfo.from_metadata(frame_num=frame_num, metadata=self.metadata)


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
        self._combined_buffer_index: Optional[List[int]] = None
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
    def combined_buffer_index(self) -> List[int]:
        """
        Get the combined buffer index.
        This is a list of unique buffer indices across all recordings.
        """
        if self._combined_buffer_index is None:
            self._combined_buffer_index = make_combined_list(
                [recording.metadata["buffer_index"].tolist() for recording in self.recordings]
            )
        return self._combined_buffer_index

    @property
    def combined_frame_num(self) -> List[int]:
        """
        Get the combined frame_num.
        This is a list of unique frame_nums across all recordings.
        """
        if self._combined_frame_num is None:
            self._combined_frame_num = make_combined_list(
                [recording.metadata["frame_num"].tolist() for recording in self.recordings]
            )
        return self._combined_frame_num

    def stitch_recordings(self) -> None:
        """
        Stitch the videos together and store the result in the combined_metadata and combined_video
        """
        stitched_writes = 0
        debug_writes = 0
        for frame_num in self.combined_frame_num:
            recording_frame_pairs = []

            for recording in self.recordings:
                if frame_num in recording.metadata["frame_num"].values:
                    frame_info = FrameInfo.from_metadata(
                        frame_num=frame_num, metadata=recording.metadata
                    )
                    recording_frame_pairs.append((recording, frame_info))

            # Build frames list for all recordings that have this frame_num
            try:
                # Collect valid (recording, frame, num_buffers, sum_black_padding) tuples
                valid_pairs: List[Tuple[RecordingData, np.ndarray, int, int]] = []
                for recording, frame_info in recording_frame_pairs:
                    frame = recording.video_reader.read_frame(frame_info.reconstructed_frame_index)
                    if frame is not None:
                        rows = recording.metadata[recording.metadata["frame_num"] == frame_num]
                        num_buffers = int(len(rows))
                        sum_black = (
                            int(rows["black_padding_px"].fillna(0).sum())
                            if "black_padding_px" in rows.columns
                            else 0
                        )
                        valid_pairs.append((recording, frame, num_buffers, sum_black))

                frames: List[np.ndarray] = [vp[1] for vp in valid_pairs]

                # Determine most proper index by metadata first: more buffers, then less padding
                most_proper_idx, candidates, is_tie = most_proper_metadata(valid_pairs)
                # If metadata is tied, break tie using image-based scoring
                if is_tie:
                    candidate_frames = [frames[i] for i in candidates]
                    rel_idx, _scores = most_proper_frame(candidate_frames)
                    most_proper_idx = candidates[int(rel_idx)]

                # If multiple recordings exist, write debug composites comparing to selected
                if len(frames) > 1:
                    base_idx = most_proper_idx
                    base = frames[base_idx]
                    others = [(i, f) for i, f in enumerate(frames) if i != base_idx]
                    if not all(np.array_equal(base, f) for (_i, f) in others):
                        for idx, frame in others:
                            if base.shape != frame.shape:
                                logger.debug(
                                    f"Frames differ for frame {frame_num}"
                                    f": shape {base.shape} vs {frame.shape}"
                                )
                                continue
                            diff_mask = (base != frame).astype(np.uint8) * 255
                            diff_pixels = int(np.count_nonzero(diff_mask))
                            logger.info(
                                f"Frames are not the same for frame {frame_num} "
                                f"(Rec {base_idx} vs Rec {idx}): {diff_pixels} px differ"
                            )

                            if self.debug_video_writer is not None:
                                try:
                                    composite = np.vstack([base, frame, diff_mask])
                                    self.debug_video_writer.write_frame(composite)
                                    debug_writes += 1
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to write composite for frame {frame_num}: {e}"
                                    )
                            # Write debug metadata row if configured
                            if self.debug_csv_writer is not None:
                                base_rec, _, base_buffers, base_black = valid_pairs[base_idx]
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
                # For one or more recordings, select one of the recordings for stitched outputs
                if len(frames) >= 1:
                    try:
                        selected_frame = frames[most_proper_idx]
                        self.combined_video_writer.write_frame(selected_frame)
                        stitched_writes += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to write stitched frame {frame_num}: {e}"
                            f" (shape={getattr(selected_frame,'shape',None)}"
                            f" dtype={getattr(selected_frame,'dtype',None)})"
                        )

                    # Append metadata rows for the selected recording
                    # Align reconstructed_frame_index
                    try:
                        selected_recording = valid_pairs[most_proper_idx][0]
                        rows = selected_recording.metadata[
                            selected_recording.metadata["frame_num"] == frame_num
                        ].copy()
                        # Align reconstructed_frame_index with stitched video index
                        rows["reconstructed_frame_index"] = self._out_frame_index
                        if self.combined_metadata is None:
                            self.combined_metadata = rows
                        else:
                            self.combined_metadata = pd.concat(
                                [self.combined_metadata, rows], ignore_index=True
                            )
                        self._out_frame_index += 1
                    except Exception as e:
                        logger.debug(f"Failed to collect metadata for frame {frame_num}: {e}")
            except Exception as e:
                logger.debug(f"Error processing frame_num {frame_num}: {e}")

        # finalize writers and csv
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
        logger.info(
            f"Stitch completed: stitched_writes={stitched_writes}, debug_writes={debug_writes}"
        )


# script run for development
if __name__ == "__main__":
    recordings = [
        RecordingData(
            video_path=Path("user_data/stitch_test/stream1.avi"),
            csv_path=Path("user_data/stitch_test/stream1.csv"),
        ),
        RecordingData(
            video_path=Path("user_data/stitch_test/stream2.avi"),
            csv_path=Path("user_data/stitch_test/stream2.csv"),
        ),
    ]
    recording_bundle = RecordingDataBundle(
        recordings=recordings,
        combined_video_writer=VideoWriter(path=Path("user_data/stitch_test/stitched.avi"), fps=20),
        debug_video_writer=VideoWriter(path=Path("user_data/stitch_test/debug.avi"), fps=20),
        combined_csv_path=Path("user_data/stitch_test/stitched.csv"),
        debug_csv_path=Path("user_data/stitch_test/debug.csv"),
    )
    # list of imported recordings (video filenames)
    logger.info(f"Imported recordings: {[recording.video_path for recording in recordings]}")
    recording_bundle.stitch_recordings()

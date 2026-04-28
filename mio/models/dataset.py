"""
A :class:`.Dataset` is the collection of data produced during one contiguous period of time
(or, informally, a "session").

It can include multiple :class:`Recording` s,
which are the data produced by a single device during the collection of the dataset.
A :class:`.Recording` may consist of several different modalities of data,
or "streams",
like a :class:`.Video`, a raw binary stream, and an accompanying metadata CSV.

The items within a recording are assumed to be of the same timebase.
If a device produces multiple streams of data in different timebases
(e.g. video and electrophysiology),
then those should be considered separate recordings.

A dataset might consist of multiple recordings from different devices that need to be aligned
(e.g. multiple cameras from multiple angles, multiple sensors receiving the same stream, etc.).
The dataset can contain an ``alignment_map`` that maps a common, contiguous, monotonic index
onto the indexes of individual recordings.

Recordings may be related to or derived from other recordings:
e.g. A video can be indicated as being derived from a binary stream,
a preprocessed, denoised, etc. video can be derived from the raw video,
and so on.
A derivation is indicated by a reference from the derived to source recording
and the transformation that was applied.

Timestamps within a dataset are assumed to be in the same unit (e.g. datetimes or unix epoch floats)
and in the same timezone,
but not necessarily entirely equivalent
(e.g. multiple machines with system clocks synchronized with NTP).

A dataset is assumed to be on disk, and only small, text-based streams are loaded into memory.
The recordings within a dataset are thus primarily represented as paths,
but provide iterators or other accessors to get their contents by slicing syntax.

.. todo:

    Currently a dataset is defined by directory structure,
    where the different streams within a recording share a filename stem but differ in extension.
    Filenames should not be interpreted as bearing metadata:
    The filename should only be used to indicate recording identity and sequential order
    (where sequential order can be indicated with an integer or a timestamp).
    The dataset structure is currently flat: nested directories will be ignored.

    The "special" dataset-level fields are indicated by special filenames,
    e.g. ``config.yaml`` , ``alignment_map.yaml`` , etc.

    In the future this will be expanded to be a declaration in a metadata file,
    allowing explicit specification and making paths arbitrary.

.. todo:

    The formalization of what counts as a "device" is WIP.

"""

import re
import sys
import warnings
from pathlib import Path
from typing import Annotated as A
from typing import Any, Literal, TypeAlias

import pandas as pd
from numpydantic import NDArraySchema
from numpydantic.interface.video import VideoProxy
from pandera.typing.pandas import DataFrame
from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)

from mio.devices.stream import StreamBufferTable
from mio.devices.tables import NoiseTable, StitchTable, TimestampTable
from mio.models import MiniscopeIOModel
from mio.models.process import NoisePatchConfig
from mio.utils import _format_ranges

if sys.version_info < (3, 11):
    from typing_extensions import Self, TypedDict
else:
    from typing import Self, TypedDict

VIDEO_EXTENSIONS = (".avi", ".mp4")
RECORDING_TYPES = Literal["raw", "stitched"]
DERIVATION_TYPES = Literal["stitched"]

VideoType: TypeAlias = (
    A[VideoProxy, NDArraySchema(("*", "*", "*"))] | A[VideoProxy, NDArraySchema(("*", "*", "*", 3))]
)


class RecordingDerivation(MiniscopeIOModel):
    """How a recording was derived from other recordings"""

    type: DERIVATION_TYPES
    sources: set[str]
    """Which other recordings this recording was derived from"""


class RecordingPaths(TypedDict):
    """Filenames for potential parts of a recording"""

    video: Path
    """{stem}.avi"""
    metadata: Path
    """{stem}.csv"""
    timestamps: Path
    """{stem}_timestamps.csv"""
    noise: Path
    """{stem}_noise.csv"""
    binary: Path
    """{stem}.bin"""


def paths_from_video(video: Path) -> RecordingPaths:
    """Given some path to a root video, create the expected paths for its components"""
    return RecordingPaths(
        video=video,
        metadata=video.with_suffix(".csv"),
        timestamps=video.with_name(video.stem + "_timestamps.csv"),
        noise=video.with_name(video.stem + "_noise.csv"),
        binary=video.with_suffix(".bin"),
    )


class Recording(MiniscopeIOModel):
    """A single set of matching data streams from a device within a dataset."""

    name: str
    """The name of the recording used in filenames to group them together"""
    type: RECORDING_TYPES
    """What type of recording this is"""
    video: VideoType
    """A video created as part of this recording"""
    metadata: DataFrame[StreamBufferTable] | None = None
    """Metadata for frames within the video"""
    timestamps: DataFrame[TimestampTable] | None = None
    """
    Timestamps table, (currently) stored as ``{video_name}_timestamps.csv`` next to the video.
    When instantiating a recording, if a metadata file exists but timestamps do not,
    they are automatically generated. 
    """
    noise: DataFrame[NoiseTable] | None = None
    """
    Framewise noise measurements (created with :meth:`score_noise` ).
    """
    binary: Path | None = None
    """Path to any raw binary version of the data in the video"""
    derived_from: RecordingDerivation | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @property
    def paths(self) -> RecordingPaths:
        """Given some video, the expected paths for its related components"""
        return paths_from_video(self.video.path)

    @classmethod
    def from_video(cls, path: Path) -> "RecordingUnion":
        """Find the adjoining files from the video path"""
        path = Path(path)

        if "stitched" in path.stem:
            return StitchedRecording.from_video(path)
        else:
            return RawVideoRecording(name=path.stem, video=path)

    def score_noise(
        self, config: NoisePatchConfig | None = None, progress: bool = False, force: bool = False
    ) -> pd.DataFrame:
        """
        Score the noise level in each frame with :func:`.score_noise`,
        saving as a csv with `{name}_noise.csv`
        """

        from mio.process.video import score_noise

        if config is None:
            config = NoisePatchConfig()
        if not force:
            if self.noise is not None:
                return self.noise
            elif self.paths["noise"].exists():
                self.noise = pd.read_csv(self.paths["noise"])
                return self.noise

        self.noise = score_noise(self, config, progress=progress)
        self.noise.to_csv(self.paths["noise"], index=False)
        return self.noise

    @model_validator(mode="before")
    @classmethod
    def _load_csvs(cls, v: dict) -> dict:
        video = v.get("video")
        video_path: Path = video.path if isinstance(video, VideoProxy) else Path(video)
        paths = paths_from_video(video_path)
        for key, path in paths.items():
            if key in v:
                continue
            elif path.suffix == ".csv" and path.exists():
                v[key] = pd.read_csv(path)
            elif path.suffix != ".csv" and path.exists():
                v[key] = path

        return v

    @model_validator(mode="after")
    def _metadata_length_matches_video(self) -> Self:
        """Video has the same number of frames as accompanying metadata"""
        if self.metadata is not None and "reconstructed_frame_index" in self.metadata:
            video_frames = set(range(self.video.shape[0]))
            metadata_frames = set(self.metadata["reconstructed_frame_index"].unique())

            # handle off-by-one error
            # https://github.com/miniscope/mio/pull/133#issuecomment-4270192079
            # lets focus on fixing the underlying bug before expanding this check beyond 1 frame.
            if len(metadata_frames) == len(video_frames) + 1:
                warnings.warn(
                    f"Metadata for {self.video.path} has an extra frame that was not "
                    "written to video, trimming loaded metadata.",
                    stacklevel=2,
                )
                self.metadata = self.metadata[
                    self.metadata["reconstructed_frame_index"].isin(video_frames)
                ]
                metadata_frames = set(self.metadata["reconstructed_frame_index"].unique())

            assert video_frames == metadata_frames, (
                f"Metadata has different number of frames than video:\n"
                f"Metadata extra: {_format_ranges(metadata_frames - video_frames)}\n"
                f"Video extra: {_format_ranges(video_frames - metadata_frames)}"
            )
        return self

    @model_validator(mode="after")
    def _ensure_timestamps(self) -> Self:
        """Ensure that timestamps are created if metadata exists but timestamps don't"""
        if self.metadata is not None and self.timestamps is None:
            from mio.process.video import _make_frame_timestamp_csv

            timestamps = _make_frame_timestamp_csv(self.metadata)
            timestamps.to_csv(self.paths["timestamps"], index=False)
            self.timestamps = timestamps
        return self


class RawVideoRecording(Recording):
    """A raw video"""

    type: Literal["raw"] = "raw"


class StitchedRecording(Recording):
    """Multiple video recordings stitched together, picking one best aligned frame from each"""

    type: Literal["stitched"] = "stitched"
    metadata: DataFrame[StreamBufferTable]
    scores: DataFrame[StitchTable]
    """A csv that indicates which recording each stitched frame was selected from"""
    derived_from: RecordingDerivation
    """A derivation reference that indicates which videos this stitch was derived from"""
    debug_video: VideoType | None = None
    """An optional debug video that shows the source videos side by side with differences marked"""

    @classmethod
    def from_video(cls, path: Path) -> "StitchedRecording":
        """Determine which videos we were derived from using the path name"""
        stem = re.sub(r"_stitched$", "", path.stem)
        sources = stem.split("__")
        if not len(sources) == 2:
            raise ValueError(
                f"Can't determine source video names from path name: {path},"
                f"the two names should be separated by __"
            )
        derived_from = RecordingDerivation(type="stitched", sources=set(sources))
        return StitchedRecording(name=path.stem, video=path, derived_from=derived_from)

    @model_validator(mode="before")
    @classmethod
    def _load_scores(cls, v: dict) -> dict:
        if v.get("scores") is not None:
            return v
        video = v.get("video")
        video_path: Path = video.path if isinstance(video, VideoProxy) else Path(video)
        scores_path = video_path.parent / (video_path.stem + "_scores.csv")
        if scores_path.exists():
            v["scores"] = pd.read_csv(scores_path)
        return v

    @model_validator(mode="before")
    @classmethod
    def _load_debug(cls, v: dict) -> dict:
        if v.get("debug_video") is not None:
            return v
        video = v.get("video")
        video_path: Path = video.path if isinstance(video, VideoProxy) else Path(video)
        debug_video_path = video_path.parent / (video_path.stem + "_debug.avi")
        if debug_video_path.exists():
            v["debug_video"] = debug_video_path
        return v


def _recording_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "raw")
    return getattr(v, "type", "raw")


RecordingUnion: TypeAlias = A[
    A[RawVideoRecording, Tag("raw")] | A[StitchedRecording, Tag("stitched")],
    Discriminator(_recording_discriminator),
]


class Dataset(MiniscopeIOModel):
    """
    A single capture from a mio device,
    including any videos, metadata tables, and other byproducts
    """

    path: Path
    """The directory where the files within the dataset are contained"""
    recordings: dict[str, Recording] = Field(default_factory=dict)
    """Recordings within this dataset"""
    alignment_map: pd.DataFrame | None = None
    """
    A dataframe with a column "index" that is the common index for frames within recordings,
    and columns for each recording name containing the index that the mapped index corresponds to
    such that each frame within a row was captured at the same time.
    
    Stored as `alignment_map.csv` in the dataset directory
    
    E.g. if a dataset contains two videos "a" and "b", and "b" started 5 frames before "a",
    then the alignment map would look like: 
    
    | index | a | b |
    | ----- | - | - |
    |     0 | 0 | 5 |
    |     1 | 1 | 6 |
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_directory(cls, path: Path) -> "Dataset":
        """Read a dataset from a directory"""
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory")

        recordings = {}
        for file in path.iterdir():
            if file.suffix in VIDEO_EXTENSIONS:
                recording = Recording.from_video(file)
                recordings[recording.name] = recording

        alignment_map = None
        if (alignment_path := path / "alignment_map.csv").exists():
            alignment_map = pd.read_csv(alignment_path)

        return cls(path=path, recordings=recordings, alignment_map=alignment_map)

    @classmethod
    def from_recordings(cls, recordings: list[Recording]) -> "Dataset":
        """
        Instantiate a dataset from recordings, loading any alignment map found.
        """
        if not all(rec.video.path.parent == recordings[0].video.path.parent for rec in recordings):
            raise ValueError(
                "All input videos should be in the same dataset, "
                "and therefore be in the same directory."
            )
        path = recordings[0].video.path.parent
        alignment_map = None
        if (alignment_path := path / "alignment_map.csv").exists():
            alignment_map = pd.read_csv(alignment_path)
        return cls(
            path=path, recordings={r.name: r for r in recordings}, alignment_map=alignment_map
        )

    def align(self, recordings: list[Recording] | list[str], write: bool = False) -> Self:
        """Create an alignment map, or return an already-existing alignment map"""
        from mio.process.stitch import align

        names: list[str] = [r.name if isinstance(r, Recording) else r for r in recordings]
        recs = [r if isinstance(r, Recording) else self.recordings[r] for r in recordings]

        if self.alignment_map is not None and all(
            name in self.alignment_map.columns for name in names
        ):
            return self
        else:
            self.alignment_map = align(recs)
            if write:
                self.alignment_map.to_csv(self.path / "alignment_map.csv", index=False)
            return self

    def stitch(self, recordings: list[Recording] | list[str]) -> Self:
        """
        Combine multiple recordings from the same device into a single recording
        by selecting the best matching frame from each.

        See :func:`~mio.process.stitch.stitch` for more details.
        """
        from mio.process.stitch import stitch

        recs = [r if isinstance(r, Recording) else self.recordings[r] for r in recordings]

        stitched = stitch(recs, dataset=self)
        self.recordings[stitched.name] = stitched
        return self

    def get_stitched(self, recordings: list[Recording] | list[str]) -> StitchedRecording:
        """
        Get a stitched recording of a set of recordings, if it exists,
        otherwise throw a KeyError
        """
        names = set([r.name if isinstance(r, Recording) else r for r in recordings])
        stitched: list[StitchedRecording] = [
            r
            for r in self.recordings.values()
            if isinstance(r, StitchedRecording) and r.derived_from.sources == names
        ]
        if len(stitched) != 0:
            return stitched[0]
        else:
            raise KeyError(
                f"No stitched recording for recordings {names} - call stitch to make one"
            )

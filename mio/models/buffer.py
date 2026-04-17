"""
Models for a data stream from a miniscope device: header formats,
containers, etc.
"""

from collections.abc import Sequence
from typing import TypeVar

from mio.models import Container, MiniscopeConfig
from mio.models.mixins import ConfigYAMLMixin


class BufferHeaderFormat(MiniscopeConfig, ConfigYAMLMixin):
    """
    Format model used to parse header at the beginning of every buffer.

    Parameters
    ----------
    linked_list: int
        Index of data buffers within the circulating structure.
        This increments with each buffer until it reaches [`num_buffers`](../api/stream.md),
        then resets to zero.
    frame_num: int
        The index of the image frame, which increments with each image frame
        (comprising multiple data buffers).
    buffer_count: int
        Index of data buffers, which increments with each buffer.
    frame_buffer_count: int
        Index of the data buffer within the image frame.
        It is set to `frame_buffer_count = 0` at the first buffer in each frame.
    write_buffer_count: int
        Number of data buffers transmitted out of the MCU.
    dropped_buffer_count: int
        Number of dropped data buffers.
    timestamp: int
        Timestamp in milliseconds.
        This should increase approximately by `1 / framerate * 1000` every frame.
    write_timestamp: int
        Timestamp in milliseconds when the buffer was transmitted out of the MCU.
    """

    linked_list: int
    frame_num: int
    buffer_count: int
    frame_buffer_count: int
    write_buffer_count: int
    dropped_buffer_count: int
    timestamp: int
    write_timestamp: int


_T = TypeVar("_T", bound="BufferHeader")


class BufferHeader(Container):
    """
    Container for the data stream's header, structured by :class:`.MetadataHeaderFormat`
    """

    linked_list: int
    frame_num: int
    buffer_count: int
    frame_buffer_count: int
    write_buffer_count: int
    dropped_buffer_count: int
    timestamp: int
    write_timestamp: int

    @classmethod
    def from_format(
        cls: type[_T], vals: Sequence, format: BufferHeaderFormat, construct: bool = False
    ) -> _T:
        """
        Instantiate a buffer header from linearized values (eg. in an ndarray or list)
        and an associated format that tells us what index the model values are found
        in that data.

        Args:
            vals (list, :class:`numpy.ndarray` ): Indexable values to cast to the header model
            format (:class:`.BufferHeaderFormat` ): Format used to index values
            construct (bool): If ``True`` , use :meth:`~pydantic.BaseModel.model_construct`
                to create the model instance (ie. without validation, but faster).
                Default: ``False``

        Returns:
            :class:`.BufferHeader`
        """

        header_data = dict()
        for hd, header_index in format.model_dump(exclude=set(format.HEADER_FIELDS)).items():
            if header_index is not None:
                header_data[hd] = vals[header_index]

        if construct:
            return cls.model_construct(**header_data)
        else:
            return cls(**header_data)

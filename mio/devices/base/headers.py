"""Base device headers"""

import sys
from collections.abc import Sequence
from typing import ClassVar

from mio.models import Container

if sys.version_info <= (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class BufferHeader(Container):
    """
    Container for the data stream's header, structured by :class:`.MetadataHeaderFormat`
    """

    POSITIONS: ClassVar[dict[str, int]] = {
        "linked_list": 0,
        "frame_num": 1,
        "buffer_count": 2,
        "frame_buffer_count": 3,
        "write_buffer_count": 4,
        "dropped_buffer_count": 5,
        "timestamp": 6,
        "write_timestamp": 7,
    }

    linked_list: int
    frame_num: int
    buffer_count: int
    frame_buffer_count: int
    write_buffer_count: int
    dropped_buffer_count: int
    timestamp: int
    write_timestamp: int

    @classmethod
    def from_sequence(cls, vals: Sequence, construct: bool = False) -> Self:
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
        for hd, header_index in cls.POSITIONS.items():
            if header_index is not None:
                header_data[hd] = vals[header_index]

        if construct:
            return cls.model_construct(**header_data)
        else:
            return cls(**header_data)

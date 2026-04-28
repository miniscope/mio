"""Base device headers"""

import sys
from collections.abc import Sequence
from typing import Any, ClassVar

import pandera.pandas as pa

from mio.models import Container, Table

if sys.version_info <= (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class BufferHeader(Container):
    """
    Container for the data stream's header.

    When converting from a sequence (like a numpy array or list of bytes),
    the ``POSITIONS`` mapping is responsible for mapping sequential values to keys,
    as different devices have the fields in different orders.

    Not all keys must be set in ``POSITIONS`` -
    extra keys may be provided e.g. at runtime while constructing the object
    by passing ``**kwargs`` to :meth:`.from_sequence`
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
    def from_sequence(
        cls, vals: Sequence, construct: bool = False, **kwargs: dict[str, Any]
    ) -> Self:
        """
        Instantiate a buffer header from linearized values (eg. in an ndarray or list)
        and an associated format that tells us what index the model values are found
        in that data.

        Args:
            vals (list, :class:`numpy.ndarray` ): Indexable values to cast to the header model
            construct (bool): If ``True`` , use :meth:`~pydantic.BaseModel.model_construct`
                to create the model instance (ie. without validation, but faster).
                Default: ``False``
            **kwargs: Additional kwargs that are not specified in ``POSITIONS`` .
                Values in ``kwargs`` overwrite those derived from ``vals`` .

        Returns:
            :class:`.BufferHeader`
        """

        header_data = dict()
        for hd, header_index in cls.POSITIONS.items():
            if header_index is not None:
                header_data[hd] = vals[header_index]

        header_data.update(kwargs)

        if construct:
            return cls.model_construct(**header_data)
        else:
            return cls(**header_data)

    @classmethod
    def csv_header_cols(cls) -> list[str]:
        """
        Return the standardized column names for CSV output.

        This ensures consistent column ordering across all StreamBufferHeader instances
        when writing to CSV files.

        Args:
            header_format: The StreamBufferHeaderFormat instance to get column ordering from

        Returns:
            list[str]: Column names in the order they should appear in CSV output
        """
        # Get the base header format columns (excluding internal fields)
        header_items = sorted(cls.POSITIONS.items(), key=lambda x: x[1])
        positioned_cols = [name for name, _ in header_items]

        other_fields = [field for field in cls.model_fields if field not in positioned_cols]

        return positioned_cols + other_fields


class BufferTable(Table):
    """Table corresponding to the BufferHeader record"""

    _RECORD_MODEL = BufferHeader

    linked_list: int = pa.Field(ge=0, coerce=True)
    frame_num: int = pa.Field(ge=0, coerce=True)
    buffer_count: int = pa.Field(ge=0, coerce=True)
    frame_buffer_count: int = pa.Field(ge=0, coerce=True)
    write_buffer_count: int = pa.Field(ge=0, coerce=True)
    dropped_buffer_count: int = pa.Field(ge=0, coerce=True)
    timestamp: int = pa.Field(ge=0, coerce=True)
    write_timestamp: int = pa.Field(ge=0, coerce=True)

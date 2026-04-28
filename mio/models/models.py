"""
Base and meta model classes.
"""

import sys
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
from pydantic import BaseModel

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class MiniscopeIOModel(BaseModel):
    """
    Root model for all mio models
    """


class MiniscopeConfig(MiniscopeIOModel):
    """
    Root model for all configuration models,
    eg. those that are effectively static at runtime.

    .. note::
        Not named ``Config`` or ``BaseConfig`` because those are both
        in use already.

    See also: :class:`.Container`
    """


class Container(MiniscopeIOModel):
    """
    Root model for models intended to be used as runtime data containers,
    eg. those that actually carry data from a buffer, rather than
    those that configure positions within a header.

    See also: :class:`.MiniscopeConfig`
    """


class Table(pa.DataFrameModel):
    """
    Root model for metadata tables.
    Each table should have a corresponding record model for its individual rows
    """

    _RECORD_MODEL: ClassVar[type[Container] | None] = None

    @classmethod
    def read_csv(cls, path: Path, **kwargs: dict[str, Any]) -> DataFrame[Self]:
        """Read and validate a table as a csv"""
        df = pd.read_csv(path, **kwargs)
        return cls.validate(df, inplace=True)

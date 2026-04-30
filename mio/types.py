"""
Type and type annotations
"""

import re
import sys
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, Field

if sys.version_info < (3, 13):
    from typing_extensions import TypeIs
else:
    from typing import TypeIs

CONFIG_ID_PATTERN = r"[\w\-\/#]+"
"""
Any alphanumeric string (\\w), as well as
- ``-``
- ``/``
- ``#``
(to allow hierarchical IDs as well as fragment IDs).

Specficially excludes ``.`` to avoid confusion between IDs, paths, and python module names

May be made less restrictive in the future, will not be made more restrictive.
"""


def _is_identifier(val: str) -> str:
    # private validation method to validate the parts of a fully-qualified python identifier
    # defined first and not made public bc used as a validator,
    # distinct from a boolean "is_{x}" check

    for part in val.split("."):
        assert part.isidentifier(), f"{part} is not a valid python identifier within {val}"
    return val


Range: TypeAlias = tuple[int, int] | tuple[float, float]
PythonIdentifier: TypeAlias = Annotated[str, AfterValidator(_is_identifier)]
"""
A valid python identifier, including globally namespace pathed like 
module.submodule.ClassName
"""
ConfigID: TypeAlias = Annotated[str, Field(pattern=CONFIG_ID_PATTERN)]
"""
A string that refers to a config file by the ``id`` field in that config
"""
ConfigSource: TypeAlias = Path | PathLike[str] | ConfigID
"""
Union of all types of config sources
"""


def valid_config_id(val: Any) -> TypeIs[ConfigID]:
    """
    Checks whether a string is a valid config id.
    """
    if not isinstance(val, str):
        return False
    return bool(re.fullmatch(CONFIG_ID_PATTERN, val))

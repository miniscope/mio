import pytest

from mio.devices.base import BufferHeader
from mio.models import Table


def _subclasses(cls: type) -> list[type]:
    subs = []
    my_subs = cls.__subclasses__()
    subs.extend(my_subs)
    for sub in my_subs:
        subs.extend(_subclasses(sub))
    return subs


@pytest.mark.parametrize("table", _subclasses(Table))
def test_table_models_match_records(table: Table):
    """All table models should match their indicated record model"""
    if table._RECORD_MODEL is None:
        return

    if issubclass(table._RECORD_MODEL, BufferHeader):
        record_fields = table._RECORD_MODEL.csv_header_cols()
    else:
        record_fields = list(table._RECORD_MODEL.model_fields.keys())

    assert record_fields == list(table.build_schema_().columns.keys())

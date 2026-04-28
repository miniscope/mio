import yaml
from pydantic import BaseModel

from mio.models.mixins import YAMLMixin


def test_yaml_mixin(tmp_path):
    """
    YAMLMixIn should give our models a from_yaml method to read from files
    """

    class MyModel(BaseModel, YAMLMixin):
        a_str: str
        a_int: int
        a_list: list[int]
        a_dict: dict[str, float]

    data = {"a_str": "string!", "a_int": 5, "a_list": [1, 2, 3], "a_dict": {"a": 1.1, "b": 2.5}}

    yaml_file = tmp_path / "temp.yaml"
    with open(yaml_file, "w") as yfile:
        yaml.safe_dump(data, yfile)

    instance = MyModel.from_yaml(yaml_file)
    assert instance.model_dump() == data

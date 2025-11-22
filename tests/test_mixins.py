from pathlib import Path
from importlib.metadata import version

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from const import CONFIG_DIR
from mio.models.mixins import yaml_peek, ConfigYAMLMixin
from tests.fixtures import tmp_config_source, yaml_config


class NestedModel(BaseModel):
    d: int = 4
    e: str = "5"
    f: float = 5.5


class MyModel(ConfigYAMLMixin):
    id: str = "my-config"
    a: int = 0
    b: str = "1"
    c: float = 2.2
    child: NestedModel = NestedModel()


class LoaderModel(ConfigYAMLMixin):
    """Model that just allows everything, only used to test write on load"""

    model_config = ConfigDict(extra="allow")


@pytest.mark.parametrize(
    "id,path,valid",
    [
        ("default-path", None, True),
        ("nested-path", Path("configs/nested/path/config.yaml"), True),
        ("not-valid", Path("not_in_dir/config.yaml"), False),
    ],
)
def test_config_from_id(yaml_config, id, path, valid):
    """Configs can be looked up with the id field if they're within a config directory"""
    instance = MyModel(id=id)
    yaml_config(id, instance.model_dump(), path)
    if valid:
        loaded = MyModel.from_id(id)
        assert loaded == instance
        assert loaded.child == instance.child
        assert isinstance(loaded.child, NestedModel)
    else:
        with pytest.raises(KeyError):
            MyModel.from_id(id)


@pytest.mark.parametrize(
    "id,id_or_path,path,valid",
    [
        ("default-path", "default-path", None, True),
        (
            "nested-path",
            Path("nested/path/config.yaml"),
            Path("configs/nested/path/config.yaml"),
            True,
        ),
        (
            "nested-path",
            "nested-path",
            Path("configs/nested/path/config.yaml"),
            True,
        ),
        ("not-valid", Path("not_in_dir/config.yaml"), Path("not_in_dir/config.yaml"), False),
    ],
)
def test_config_from_any(yaml_config, id, id_or_path, path, valid):
    """
    Configs can be looked up with a path or an id field if they're within a config directory
    """
    instance = MyModel(id=id)
    abs_path = yaml_config(id, instance.model_dump(), path)
    if valid:
        loaded = MyModel.from_any(id_or_path)
        assert loaded == instance
        assert loaded.child == instance.child
        assert isinstance(loaded.child, NestedModel)
    else:
        with pytest.raises((KeyError, ValueError)):
            MyModel.from_any(id_or_path)

    # and we should always be able to load an absolute path
    _ = MyModel.from_yaml(abs_path)


def test_config_sources():
    """
    config_sources should return a list of paths
    """
    paths = ConfigYAMLMixin.config_sources()
    assert len(paths) > 0
    assert all([isinstance(p, Path) for p in paths])


def test_roundtrip_to_from_yaml(tmp_config_source):
    """Config models can roundtrip to and from yaml"""
    yaml_file = tmp_config_source / "test_config.yaml"

    instance = MyModel()
    instance.to_yaml(yaml_file)
    loaded = MyModel.from_yaml(yaml_file)
    assert loaded == instance
    assert loaded.child == instance.child
    assert isinstance(loaded.child, NestedModel)


@pytest.mark.parametrize(
    "src",
    [
        pytest.param(
            """
a: 9
b: "10\"""",
            id="missing",
        ),
        pytest.param(
            f"""
a: 9
id: "my-config"
mio_model: "tests.test_mixins.MyModel"
mio_version: "{version('mio')}"
b: "10\"""",
            id="not-at-start",
        ),
        pytest.param(
            f"""
mio_version: "{version('mio')}"
mio_model: "tests.test_mixins.MyModel"
id: "my-config"
a: 9
b: "10\"""",
            id="out-of-order",
        ),
    ],
)
def test_complete_header(tmp_config_source, src: str):
    """
    Config models saved without header information will have it filled in
    the source yaml they were loaded from
    """
    yaml_file = tmp_config_source / "test_config.yaml"

    with open(yaml_file, "w") as yfile:
        yfile.write(src)

    _ = MyModel.from_yaml(yaml_file)

    with open(yaml_file, "r") as yfile:
        loaded = yaml.safe_load(yfile)

    loaded_str = yaml_file.read_text()

    assert loaded["mio_version"] == version("mio")
    assert loaded["id"] == "my-config"
    assert loaded["mio_model"] == MyModel._model_name()

    # the header should come at the top!
    lines = loaded_str.splitlines()
    for i, key in enumerate(("id", "mio_model", "mio_version")):
        line_key = lines[i].split(":")[0].strip()
        assert line_key == key


@pytest.mark.parametrize("config_file", CONFIG_DIR.rglob("*.y*ml"))
def test_builtins_unchanged(config_file):
    """None of the builtin configs should be modified on load - i.e. they should all have correct headers."""
    before = config_file.read_text()
    _ = LoaderModel.from_yaml(config_file)
    after = config_file.read_text()
    assert (
        before == after
    ), f"Packaged config {config_file} was modified on load, ensure it has the correct headers."


@pytest.mark.parametrize(
    "key,expected,root,first",
    [
        ("key1", "val1", True, True),
        ("key1", "val1", False, True),
        ("key1", ["val1"], True, False),
        ("key1", ["val1", "val2"], False, False),
        ("key2", "val2", True, True),
        ("key3", False, True, True),
        ("key4", False, True, True),
        ("key4", "val4", False, True),
    ],
)
def test_peek_yaml(key, expected, root, first, yaml_config):
    yaml_file = yaml_config(
        "test", {"key1": "val1", "key2": "val2", "key3": {"key1": "val2", "key4": "val4"}}, None
    )

    if not expected:
        with pytest.raises(KeyError):
            _ = yaml_peek(key, yaml_file, root=root, first=first)
    else:
        assert yaml_peek(key, yaml_file, root=root, first=first) == expected

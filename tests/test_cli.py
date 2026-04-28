import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import BaseModel

from mio import Config
from mio.cli.config import _list, config, config_path, create
from mio.cli.stream import capture
from mio.models import config as _config_mod
from mio.utils import hash_video

from .conftest import CONFIG_DIR, DATA_DIR


@pytest.mark.skip("Needs to be implemented")
def test_cli_stream():
    """should be able to invoke streamdaq, using various capture options"""
    pass


def test_cli_config_show():
    """
    `mio config` should show current config
    """
    runner = CliRunner()
    result = runner.invoke(config)
    cfg_yaml = Config().to_yaml()
    assert cfg_yaml in result.output


def test_cli_config_show_global():
    """
    `mio config global` should show contents of the global config file
    """
    runner = CliRunner()
    result = runner.invoke(config, ["global"])
    cfg_yaml = _config_mod._global_config_path.read_text()
    assert str(_config_mod._global_config_path) in result.output
    assert cfg_yaml in result.output


def test_cli_config_global_path():
    """
    `mio global path` should show the path to the global config file
    """
    runner = CliRunner()
    result = runner.invoke(config, ["global", "path"])
    assert str(_config_mod._global_config_path) in result.output


def test_cli_config_user_show(set_user_yaml):
    """
    `mio config user` should show contents of the user config file
    """
    user_yaml_path = set_user_yaml({"logs": {"level": "WARNING"}})
    runner = CliRunner()
    result = runner.invoke(config, ["user"])
    user_config = user_yaml_path.read_text()
    assert "level: WARNING" in user_config
    assert user_config in result.output


@pytest.mark.parametrize("clean", [True, False])
@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_config_user_create(clean, dry_run, tmp_path):
    """
    `mio config user create` creates a new user config file,
    optionally with clean/dirty mode or dry_run or not
    """
    dry_run_cmd = "--dry-run" if dry_run else "--no-dry-run"
    clean_cmd = "--clean" if clean else "--dirty"

    config_path = tmp_path / "mio_config.yaml"

    runner = CliRunner()
    result = runner.invoke(config, ["user", "create", dry_run_cmd, clean_cmd, str(config_path)])

    if dry_run:
        assert "DRY RUN" in result.output
        assert not config_path.exists()
    else:
        assert "DRY RUN" not in result.output
        assert config_path.exists()

    if clean:
        assert "level" not in result.output
    else:
        assert "level" in result.output

    assert f"user_dir: {str(config_path.parent)}" in result.output


def test_cli_config_user_path(set_env, set_user_yaml):
    """
    `mio config user path` should show the path to the user config file
    """
    user_config_path = set_user_yaml({"logs": {"level": "WARNING"}})

    runner = CliRunner()
    result = runner.invoke(config, ["user", "path"])
    assert str(user_config_path) in result.output


def test_cli_config_list():
    """
    mio config list should list all the configs in the user directory and provided by mio
    """
    runner = CliRunner()
    result = runner.invoke(_list, color=False)

    # not testing for the literal table structure,
    # but we should have headers and some table characters
    for header_substr in ("id", "mio_model", "path"):
        assert header_substr in result.output

    if sys.platform == "win32":
        assert "\u2500" in result.output
    else:
        assert "━━" in result.output

    # configs from the temporarily configured test config directory should be included
    assert "test-wireless-200px" in result.output

    # and configs provided by mio
    assert "wirefree-sd-layout" in result.output

    # by default paths and mio models should be truncated
    if sys.platform == "win32":
        assert "\u2502 .sdcard.SDLayout" in result.output
        assert "\u2502 wirefree" in result.output
    else:
        assert "│ .sdcard.SDLayout" in result.output
        assert "│ wirefree/" in result.output

    # verbose should display the full values
    # (though truncated in testing because console width is 80)
    result = runner.invoke(_list, ["-v"], color=False)
    assert "mio.models." in result.output
    assert str(CONFIG_DIR)[0:5] in result.output


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "freq_mask_config, video_hash",
    [
        (None, "22fa7249faffff45f5f5aa12d36399da9e8be2f7c578ca2a3c7dccbaddc9063e"),
    ],
)
def test_cli_capture(
    freq_mask_config,
    video_hash: str,
    tmp_path,
    set_okdev_input,
):
    """
    Basic regression test to ensure that we can in fact call the capture cli method,
    even though it's just a wrapper of the capture method.
    """
    runner = CliRunner()
    path_stem = tmp_path / "data"
    data_file = DATA_DIR / "stream_daq_test_fpga_raw_input_200px.bin"
    set_okdev_input(data_file)
    args = ["--config", "test-wireless-200px", "--output", str(path_stem), "--no-display"]

    # bit of a ghost parameterization -
    # left as placeholder in case we want to test freq mask display
    if freq_mask_config:
        args.append("--freq_mask_config")
        args.append(freq_mask_config)

    result = runner.invoke(capture, args)
    assert result.exit_code == 0
    output_hash = hash_video(path_stem.with_suffix(".avi"))
    assert output_hash == video_hash


def test_cli_config_create_list():
    """
    mio config create --list displays a list of available models
    """
    runner = CliRunner()
    result = runner.invoke(create, ["--list"], color=False)
    assert result.exit_code == 0
    # simple test for presence, we assume that rich handles the table formatting properly

    stdout = result.stdout
    from mio.models.mixins import ConfigYAMLMixin

    for model_name, model in ConfigYAMLMixin.config_models().items():
        assert model_name in stdout
        assert f"{model.__module__}.{model.__name__}" in stdout


def test_cli_config_create(tmp_config_dir):
    """
    Create a config using the cli by passing kwargs, which should be evaluated as python literals
    """
    from mio.models.mixins import ConfigYAMLMixin

    class SubModel(BaseModel):
        a: str
        b: int

    class MyConfigModel(ConfigYAMLMixin):
        a_string: str
        a_int: int
        a_dict: SubModel

    runner = CliRunner()
    result = runner.invoke(
        create,
        [
            "MyConfigModel",
            "my-cool-config",
            "-v",
            "a_string=hey",
            "-v",
            "a_int=2",
            "-v",
            "a_dict={'a': 'a', 'b': 5}",
        ],
        color=False,
    )
    assert result.exit_code == 0
    expected_path = tmp_config_dir / "my-cool-config.yaml"
    assert expected_path.exists()

    # get raw from file, and asset matches that loaded from the id
    with open(expected_path) as f:
        data = yaml.safe_load(f)
    assert data["id"] == "my-cool-config"
    loaded = MyConfigModel(**data)
    from_id = MyConfigModel.from_id("my-cool-config")
    assert loaded == from_id


def test_cli_config_path(tmp_config_dir):
    """
    A config's path is printed from its ID
    """
    cfg_path = tmp_config_dir / "my-custom-config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump({"id": "my-custom-config", "mio_model": "mio.testing.FakeModel"}, f)

    runner = CliRunner()
    result = runner.invoke(config_path, ["my-custom-config"])
    assert result.exit_code == 0
    assert Path(result.stdout.strip()) == cfg_path

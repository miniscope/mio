from pathlib import Path
from typing import Callable, Optional, Any, MutableMapping

import pytest
import yaml
import tomli_w
from _pytest.monkeypatch import MonkeyPatch

from mio import Config
from mio.io import SDCard
from mio.models.config import _global_config_path, set_user_dir
from mio.models.frames import SDCardFrame
from mio.models.mixins import ConfigYAMLMixin, YamlDumper


@pytest.fixture
def wirefree() -> SDCard:
    """
    SDCard with wirefree layout pointing to the sample data file

    """
    sd_path = Path(__file__).parent.parent / "data" / "wirefree_example.img"
    sdcard = SDCard(drive=sd_path, layout="wirefree-sd-layout")
    return sdcard


@pytest.fixture
def wirefree_battery() -> SDCard:
    sd_path = Path(__file__).parent.parent / "data" / "wirefree_battery_sample.img"
    sdcard = SDCard(drive=sd_path, layout="wirefree-sd-layout-battery")
    return sdcard


@pytest.fixture()
def wirefree_frames(wirefree) -> SDCardFrame:
    frames = []
    with wirefree:
        while True:
            try:
                frame_object = wirefree.read(return_header=True)
                frames.append(frame_object)
            except StopIteration:
                break
    return SDCardFrame(frames=frames)


@pytest.fixture()
def tmp_config_source(tmp_path, monkeypatch) -> Path:
    """
    Monkeypatch the config sources to include a temporary path
    """

    path = tmp_path / "configs"
    path.mkdir(exist_ok=True)
    current_sources = ConfigYAMLMixin.config_sources()

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        nonlocal current_sources
        return [path, *current_sources]

    monkeypatch.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))
    return path


@pytest.fixture()
def yaml_config(
    tmp_config_source, tmp_path, monkeypatch
) -> Callable[[str, dict, Optional[Path]], Path]:
    out_file = tmp_config_source / "test_config.yaml"

    def _yaml_config(id: str, data: dict, path: Optional[Path] = None) -> Path:
        if path is None:
            path = out_file
        else:
            path = Path(path)
            if not path.is_absolute():
                # put under tmp_path (rather than tmp_config_source)
                # in case putting a file outside the config dir is intentional.
                path = tmp_path / path

            if path.is_dir():
                path.mkdir(exist_ok=True, parents=True)
                path = path / "test_config.yaml"
            else:
                path.parent.mkdir(exist_ok=True, parents=True)

        data = {"id": id, **data}
        with open(path, "w") as yfile:
            yaml.dump(data, yfile)
        return path

    return _yaml_config


@pytest.fixture(scope="session")
def monkeypatch_session() -> MonkeyPatch:
    """
    Monkeypatch you can use at the session scope!
    """
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="session", autouse=True)
def dodge_existing_configs(tmp_path_factory):
    """
    Suspend any existing global config file during config tests
    """
    tmp_path = tmp_path_factory.mktemp("config_backup")
    global_config_path = _global_config_path
    backup_global_config_path = tmp_path / "mio_config.yaml.global.bak"

    user_config_path = list(Config().user_dir.glob("mio_config.*"))
    if len(user_config_path) == 0:
        user_config_path = None
    else:
        user_config_path = user_config_path[0]

    backup_user_config_path = tmp_path / "mio_config.yaml.user.bak"

    dotenv_path = Path(".env").resolve()
    dotenv_backup_path = tmp_path / "dotenv.bak"

    if global_config_path.exists():
        global_config_path.rename(backup_global_config_path)
    if user_config_path is not None and user_config_path.exists():
        user_config_path.rename(backup_user_config_path)
    if dotenv_path.exists():
        dotenv_path.rename(dotenv_backup_path)

    yield

    if backup_global_config_path.exists():
        global_config_path.unlink(missing_ok=True)
        backup_global_config_path.rename(global_config_path)
    if backup_user_config_path.exists():
        user_config_path.unlink(missing_ok=True)
        backup_user_config_path.rename(user_config_path)
    if dotenv_backup_path.exists():
        dotenv_path.unlink(missing_ok=True)
        dotenv_backup_path.rename(dotenv_path)


@pytest.fixture()
def tmp_cwd(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def set_env(monkeypatch) -> Callable[[dict[str, Any]], None]:
    """
    Function fixture to set environment variables using a nested dict
    matching a GlobalConfig.model_dump()
    """

    def _set_env(config: dict[str, Any]) -> None:
        for key, value in _flatten(config).items():
            key = "MIO_" + key.upper()
            monkeypatch.setenv(key, str(value))

    return _set_env


@pytest.fixture()
def set_dotenv(tmp_cwd) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a .env file
    """
    dotenv_path = tmp_cwd / ".env"

    def _set_dotenv(config: dict[str, Any]) -> Path:
        with open(dotenv_path, "w") as dfile:
            for key, value in _flatten(config).items():
                key = "MIO_" + key.upper()
                dfile.write(f"{key}={value}\n")
        return dotenv_path

    return _set_dotenv


@pytest.fixture()
def set_pyproject(tmp_cwd) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a pyproject.toml file
    """
    toml_path = tmp_cwd / "pyproject.toml"

    def _set_pyproject(config: dict[str, Any]) -> Path:
        config = {"tool": {"mio": {"config": config}}}

        with open(toml_path, "wb") as tfile:
            tomli_w.dump(config, tfile)

        return toml_path

    return _set_pyproject


@pytest.fixture()
def set_local_yaml(tmp_cwd) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a mio_config.yaml file in the current directory
    """
    yaml_path = tmp_cwd / "mio_config.yaml"

    def _set_local_yaml(config: dict[str, Any]) -> Path:
        with open(yaml_path, "w") as yfile:
            yaml.dump(config, yfile, Dumper=YamlDumper)
        return yaml_path

    return _set_local_yaml


@pytest.fixture()
def set_user_yaml(tmp_path) -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to set config variables in a user config file
    """
    yaml_path = tmp_path / "user" / "mio_config.yaml"
    yaml_path.parent.mkdir(exist_ok=True)

    def _set_user_yaml(config: dict[str, Any]) -> Path:
        with open(yaml_path, "w") as yfile:
            yaml.dump(config, yfile, Dumper=YamlDumper)
        set_user_dir(yaml_path.parent)
        return yaml_path

    yield _set_user_yaml

    _global_config_path.unlink(missing_ok=True)


@pytest.fixture()
def set_global_yaml() -> Callable[[dict[str, Any]], Path]:
    """
    Function fixture to reversibly set config variables in a global mio_config.yaml file
    """

    def _set_global_yaml(config: dict[str, Any]) -> Path:
        with open(_global_config_path, "w") as gfile:
            yaml.dump(config, gfile, Dumper=YamlDumper)
        return _global_config_path

    yield _set_global_yaml

    _global_config_path.unlink(missing_ok=True)


@pytest.fixture(
    params=[
        "set_env",
        "set_dotenv",
        "set_pyproject",
        "set_local_yaml",
        "set_user_yaml",
        "set_global_yaml",
    ]
)
def set_config(request) -> Callable[[dict[str, Any]], Path]:
    return request.getfixturevalue(request.param)


def _flatten(d, parent_key="", separator="__") -> dict:
    """https://stackoverflow.com/a/6027615/13113166"""
    items = []
    for key, value in d.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(_flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

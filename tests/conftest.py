import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from _pytest.monkeypatch import MonkeyPatch

from mio.models.mixins import ConfigYAMLMixin

from .fixtures import *

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = DATA_DIR / "config"
MOCK_DIR = Path(__file__).parent / "mock"


@pytest.fixture(autouse=True)
def mock_okdev(monkeypatch: MonkeyPatch) -> None:
    from mio.devices.stream import nodes
    from mio.interfaces import opalkelly
    from mio.interfaces.mocks import okDevMock

    monkeypatch.setattr(opalkelly, "okDev", okDevMock)
    monkeypatch.setattr(nodes, "okDev", okDevMock)


@pytest.fixture(scope="session", autouse=True)
def mock_config_source(monkeypatch_session: MonkeyPatch) -> None:
    """
    Add the `tests/data/config` directory to the config sources for the entire testing session
    """
    current_sources = ConfigYAMLMixin.config_sources()

    def _config_sources(cls: type[ConfigYAMLMixin]) -> list[Path]:
        nonlocal current_sources
        return [CONFIG_DIR, *current_sources]

    monkeypatch_session.setattr(ConfigYAMLMixin, "config_sources", classmethod(_config_sources))


@pytest.fixture(scope="session", autouse=True)
def set_matplotlib_backend() -> None:
    """Use headless agg backend during tests"""
    import matplotlib

    matplotlib.use("agg")


@pytest.fixture()
def set_okdev_input(monkeypatch: MonkeyPatch) -> Callable[[str | Path], None]:
    """
    closure fixture to set the environment variable used by StreamDevice to set the
    okDev data source
    """

    def _set_okdev_input(file: str | Path) -> None:
        from mio.interfaces.mocks import okDevMock

        monkeypatch.setattr(okDevMock, "DATA_FILE", file)
        os.environ["PYTEST_OKDEV_DATA_FILE"] = str(file)

    return _set_okdev_input


@pytest.fixture()
def config_override(tmp_path: Path) -> Callable[[Path, dict], Path]:
    """
    Create a config file with some of its properties overridden
    """

    def _config_override(path: Path, config: dict) -> Path:
        with open(path) as f:
            data = yaml.safe_load(f)
        data.update(config)
        out_path = tmp_path / f"config_override_{datetime.now().strftime('%H_%M_%S_%f')}.yml"
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f)
        return out_path

    yield _config_override

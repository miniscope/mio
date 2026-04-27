"""
Module-global configuration models
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from platformdirs import PlatformDirs
from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from mio.models import MiniscopeIOModel
from mio.models.mixins import YAMLMixin

_default_userdir = Path().home() / ".config" / "mio"
_dirs = PlatformDirs("mio", "mio")
_global_config_path = Path(_dirs.user_config_path) / "mio_config.yaml"
LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LogConfig(MiniscopeIOModel):
    """
    Configuration for logging
    """

    level: LOG_LEVELS = "INFO"
    """
    Severity of log messages to process.
    """
    level_file: LOG_LEVELS | None = None
    """
    Severity for file-based logging. If unset, use ``level``
    """
    level_stdout: LOG_LEVELS | None = None
    """
    Severity for stream-based logging. If unset, use ``level``
    """
    file_n: int = 5
    """
    Number of log files to rotate through
    """
    file_size: int = 2**22  # roughly 4MB
    """
    Maximum size of log files (bytes)
    """

    @field_validator("level", "level_file", "level_stdout", mode="before")
    @classmethod
    def uppercase_levels(cls, value: str | None = None) -> str | None:
        """
        Ensure log level strings are uppercased
        """
        if value is not None:
            value = value.upper()
        return value


class Config(BaseSettings, YAMLMixin):
    """
    Runtime configuration for mio.

    See https://docs.pydantic.dev/latest/concepts/pydantic_settings/

    Set values either in an ``.env`` file or using environment variables
    prefixed with ``MIO_*``. Values in nested models are separated with ``__`` ,
    eg. ``MIO_LOGS__LEVEL``

    See ``.env.example`` in repository root

    Paths are set relative to ``user_dir`` by default, unless explicitly specified.
    """

    user_dir: Path = Field(
        _global_config_path.parent,
        description="Base directory to store user configuration and other temporary files, "
        "other paths are relative to this by default",
    )
    config_dir: Path = Field(Path("config"), description="Location to store user configs")
    log_dir: Path = Field(Path("logs"), description="Location to store logs")

    logs: LogConfig = Field(LogConfig(), description="Additional settings for logs")

    @field_validator("user_dir", mode="before")
    @classmethod
    def folder_exists(cls, v: Path) -> Path:
        """Ensure user_dir exists, make it otherwise"""
        v = Path(v)
        v.mkdir(exist_ok=True, parents=True)

        assert v.exists(), f"{v} does not exist!"
        return v

    @model_validator(mode="after")
    def paths_relative_to_basedir(self) -> "Config":
        """If relative paths are given, make them absolute relative to ``user_dir``"""
        paths = ("log_dir", "config_dir")
        for path_name in paths:
            path = getattr(self, path_name)  # type: Path
            if not path.is_absolute():
                path = self.user_dir / path
                setattr(self, path_name, path)
            path.mkdir(exist_ok=True)
            assert path.exists()
        return self

    model_config = SettingsConfigDict(
        env_prefix="mio_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
        yaml_file="mio_config.yaml",
        pyproject_toml_table_header=("tool", "mio", "config"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Read config settings in order of priority (high to low).

        - Arguments passed to the class constructor (not user configurable)
        - Environment variables like ``export MIO_LOG_DIR=~/``
        - ``.env`` file in the working directory
        - ``mio_config.yaml`` file in the working directory
        - ``tool.mio.config`` table in ``pyproject.toml`` in the working directory
        - User ``mio_config.yaml`` file, configured by ``user_dir``
        - Global ``mio_config.yaml`` (use ``mio config get global_config`` to find location)
        - Default values in the :class:`.GlobalConfig` model
        """
        _create_default_global_config()

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls),
            _UserYamlConfigSource(settings_cls),
            YamlConfigSettingsSource(settings_cls, yaml_file=_global_config_path),
        )


def set_user_dir(path: Path) -> None:
    """
    Set the location of the user dir in the global config file
    """
    _update_value(_global_config_path, "user_dir", str(path))


def _create_default_global_config(path: Path = _global_config_path, force: bool = False) -> None:
    """
    Create a default global `mio_config.yaml` file.

    Args:
        force (bool): Override any existing global config
    """
    if path.exists() and not force:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"user_dir": str(path.parent)}
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


class _UserYamlConfigSource(YamlConfigSettingsSource):
    """
    Yaml config source that gets the location of the user settings file from the prior sources
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._user_config = None
        super().__init__(*args, **kwargs)

    @property
    def user_config_path(self) -> Path | None:
        """
        Location of the user-level ``mio_config.yaml`` file,
        given the current state of prior config sources,
        including the global config file
        """
        config_file = None
        user_dir: str | None = self.current_state.get("user_dir", None)
        if user_dir is None:
            # try and get from global config
            if _global_config_path.exists():
                with open(_global_config_path) as f:
                    data = yaml.safe_load(f)
                user_dir = data.get("user_dir", None)

                if user_dir is not None:
                    # handle .yml or .yaml
                    config_files = list(Path(user_dir).glob("mio_config.*"))
                    if len(config_files) != 0:
                        config_file = config_files[0]

        else:
            # gotten from higher priority config sources
            config_file = Path(user_dir) / "mio_config.yaml"
        return config_file

    @property
    def user_config(self) -> dict[str, Any]:
        """
        Contents of the user config file
        """
        if self._user_config is None:
            if self.user_config_path is None or not self.user_config_path.exists():
                self._user_config = {}
            else:
                self._user_config = self._read_files(self.user_config_path)

        return self._user_config

    def __call__(self) -> dict[str, Any]:
        return (
            TypeAdapter(dict[str, Any]).dump_python(self.user_config)
            if self.nested_model_default_partial_update
            else self.user_config
        )


def _update_value(path: Path, key: str, value: Any) -> None:
    """
    Update a single value in a yaml file

    .. todo::

        Make this work with nested keys

    """
    data = None
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)

    if data is None:
        data = {}

    data[key] = value

    with open(path, "w") as f:
        yaml.dump(data, f)

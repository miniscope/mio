import os
from pathlib import Path

import numpy as np
import yaml

from mio.models.config import Config, _global_config_path, set_user_dir


def test_config(tmp_path):
    """
    Config should be able to make directories and set sensible defaults
    """
    config = Config(user_dir=tmp_path)
    assert config.user_dir.exists()
    assert config.log_dir.exists()
    assert config.log_dir == config.user_dir / "logs"


def test_set_config(set_config, tmp_path):
    """We should be able to set parameters from all available modalities"""
    file_n = int(np.random.default_rng().integers(0, 100))
    user_dir = tmp_path / f"fake/dir/{np.random.default_rng().integers(0, 100)}"

    set_config({"user_dir": str(user_dir), "logs": {"file_n": file_n}})

    config = Config()
    assert config.user_dir == user_dir
    assert config.logs.file_n == file_n


def test_config_from_environment(tmp_path):
    """
    Setting environmental variables should set the config, including recursive models
    """
    os.environ["MIO_USER_DIR"] = str(tmp_path)
    # we can also override the default log dir name
    override_logdir = Path(tmp_path) / "fancylogdir"
    os.environ["MIO_LOG_DIR"] = str(override_logdir)
    # and also recursive models
    os.environ["MIO_LOGS__LEVEL"] = "error"

    config = Config()
    assert config.user_dir == Path(tmp_path)
    assert config.log_dir == override_logdir
    assert config.logs.level == "error".upper()
    del os.environ["MIO_USER_DIR"]
    del os.environ["MIO_LOG_DIR"]
    del os.environ["MIO_LOGS__LEVEL"]


def test_config_from_dotenv(tmp_path):
    """
    dotenv files should also set config

    this test can be more relaxed since its basically a repetition of previous
    """
    tmp_path.mkdir(exist_ok=True, parents=True)
    dotenv = tmp_path / ".env"
    with open(dotenv, "w") as denvfile:
        denvfile.write(f"MIO_USER_DIR={str(tmp_path)}")

    config = Config(_env_file=dotenv, _env_file_encoding="utf-8")
    assert config.user_dir == Path(tmp_path)


def test_set_user_dir(tmp_path):
    """
    We should be able to set the user dir and the global config should respect it
    """
    user_config = tmp_path / "mio_config.yml"
    file_n = int(np.random.default_rng().integers(0, 100))
    with open(user_config, "w") as yfile:
        yaml.dump({"logs": {"file_n": file_n}}, yfile)

    set_user_dir(tmp_path)

    with open(_global_config_path) as gfile:
        global_config = yaml.safe_load(gfile)

    assert global_config["user_dir"] == str(tmp_path)
    assert Config().user_dir == tmp_path
    assert Config().logs.file_n == file_n

    # we do this manual cleanup here and not in a fixture because we are testing
    # that the things we are doing in the fixtures are working correctly!
    _global_config_path.unlink(missing_ok=True)


def test_config_sources_overrides(
    set_env, set_dotenv, set_pyproject, set_local_yaml, set_user_yaml, set_global_yaml
):
    """Test that the different config sources are overridden in the correct order"""
    set_global_yaml({"logs": {"file_n": 0}})
    assert Config().logs.file_n == 0
    set_user_yaml({"logs": {"file_n": 1}})
    assert Config().logs.file_n == 1
    set_pyproject({"logs": {"file_n": 2}})
    assert Config().logs.file_n == 2
    set_local_yaml({"logs": {"file_n": 3}})
    assert Config().logs.file_n == 3
    set_dotenv({"logs": {"file_n": 4}})
    assert Config().logs.file_n == 4
    set_env({"logs": {"file_n": 5}})
    assert Config().logs.file_n == 5
    assert Config(**{"logs": {"file_n": 6}}).logs.file_n == 6

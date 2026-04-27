import logging

import pytest
from pathlib import Path
import re
import multiprocessing as mp
from time import sleep
import warnings

from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler

from mio.logging import init_logger


@pytest.fixture(autouse=True)
def reset_root_logger():
    """
    Before each test, reset the root logger
    """
    root_logger = logging.getLogger("mio")
    root_logger.handlers.clear()


def test_init_logger(capsys, tmp_path):
    """
    We should be able to
    - log to file and stdout
    - with separable levels
    """

    log_dir = Path(tmp_path) / "logs"
    log_dir.mkdir()
    log_file = log_dir / "mio.log"
    logger = init_logger(name="test_logger", log_dir=log_dir, level="INFO", file_level="WARNING")
    warn_msg = "Both loggers should show"
    logger.warning(warn_msg)

    # can't test for presence of string because logger can split lines depending on size of console
    # but there should be one WARNING in stdout
    captured = capsys.readouterr()
    assert "WARNING" in captured.out

    with open(log_file, "r") as lfile:
        log_str = lfile.read()
    assert "WARNING" in log_str

    info_msg = "Now only stdout should show"
    logger.info(info_msg)
    captured = capsys.readouterr()
    assert "INFO" in captured.out
    with open(log_file, "r") as lfile:
        log_str = lfile.read()
    assert "INFO" not in log_str


def test_nested_loggers(capsys, tmp_path):
    """
    Nested loggers should not double-log
    """
    log_dir = Path(tmp_path)

    parent = init_logger("parent", log_dir=log_dir, level="DEBUG", file_level="DEBUG")
    child = init_logger("parent.child", log_dir=log_dir, level="DEBUG", file_level="DEBUG")

    child.debug("hey")
    parent.debug("sup")

    root_logger = logging.getLogger("mio")

    warnings.warn(f"FILES IN LOG DIR: {list(log_dir.glob('*'))}")
    warnings.warn(f"ROOT LOGGER HANDLERS: {root_logger.handlers}")

    assert len(root_logger.handlers) == 2
    assert len(parent.handlers) == 0
    assert len(child.handlers) == 0

    with open(log_dir / "mio.log") as lfile:
        file_logs = lfile.read()

    # only one message of each!
    stdout = capsys.readouterr()
    assert len(re.findall("hey", stdout.out)) == 1
    assert len(re.findall("sup", stdout.out)) == 1
    assert len(re.findall("hey", file_logs)) == 1
    assert len(re.findall("sup", file_logs)) == 1


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
@pytest.mark.parametrize("direct_setting", [True, False])
@pytest.mark.parametrize("test_target", ["logger", "RotatingFileHandler", "RichHandler"])
def test_init_logger_from_config(
    tmp_path, monkeypatch, level, direct_setting, test_target, set_config
):
    """
    Set log levels from all kinds of config
    """
    # Feels kind of fragile to hardcode this but I couldn't think of a better way so for now
    level_name_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }

    if direct_setting:
        set_config({"logs": {"level_file": level, "level_stdout": level}})
    else:
        set_config({"logs": {"level": level}})

    dotenv_logger = init_logger(name="test_logger", log_dir=tmp_path)
    root_logger = logging.getLogger("mio")

    # Separating them for readable summary info
    if test_target == "logger":
        assert dotenv_logger.level == level_name_map.get(level)

    assert len(dotenv_logger.handlers) == 0
    assert len(root_logger.handlers) == 2
    file_handlers = [h for h in root_logger.handlers if isinstance(h, RotatingFileHandler)]
    stream_handlers = [h for h in root_logger.handlers if isinstance(h, RichHandler)]
    assert len(file_handlers) == 1
    assert len(stream_handlers) == 1
    file_handler = file_handlers[0]
    stream_handler = stream_handlers[0]

    if test_target == "RotatingFileHandler":
        assert file_handler.level == level_name_map.get(level)

    elif test_target == "RichHandler":
        # Might be better to explicitly set the level in the handler
        assert stream_handler.level == level_name_map.get(level)


def _mp_function(name, path):
    logger = init_logger(name, log_dir=path, level="DEBUG", file_level="DEBUG")
    for i in range(100):
        sleep(0.001)
        logger.debug(f"{name} - {i}")


def test_multiprocess_logging(capfd, tmp_path):
    """
    We should be able to handle logging from multiple processes
    """
    # 3.14 changes default to forkserver,
    # use `spawn` explicitly until we update the logging to match noob's new version
    ctx = mp.get_context("spawn")

    proc_1 = ctx.Process(target=_mp_function, args=("proc_1", tmp_path))
    proc_2 = ctx.Process(target=_mp_function, args=("proc_2", tmp_path))
    proc_3 = ctx.Process(target=_mp_function, args=("proc_1.proc_3", tmp_path))

    proc_1.start()
    proc_2.start()
    proc_3.start()
    proc_1.join()
    proc_2.join()
    proc_3.join()

    stdout = capfd.readouterr()
    logs = {}
    for log_file in tmp_path.glob("*.log"):
        with open(log_file) as lfile:
            logs[log_file.name] = lfile.read()

    assert "mio.log" in logs
    assert len(logs) == 4

    for logfile, logs in logs.items():

        # main logfile does not receive messages
        if logfile == "mio.log":
            assert len(logs.split("\n")) == 1
        else:
            assert len(logs.split("\n")) == 101

    assert len(re.findall("DEBUG", stdout.out)) == 300

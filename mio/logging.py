"""
Logging factory and handlers
"""

import logging
import multiprocessing as mp
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from mio.models.config import LOG_LEVELS, get_config


def init_logger(
    name: str,
    log_dir: Path | None | bool = None,
    level: LOG_LEVELS | None = None,
    file_level: LOG_LEVELS | None = None,
    log_file_n: int | None = None,
    log_file_size: int | None = None,
) -> logging.Logger:
    """
    Make a logger.

    Log to a set of rotating files in the ``log_dir`` according to ``name`` ,
    as well as using the :class:`~rich.RichHandler` for pretty-formatted stdout logs.

    Args:
        name (str): Name of this logger. Ideally names are hierarchical
            and indicate what they are logging for, eg. ``mio.sdcard``
            and don't contain metadata like timestamps, etc. (which are in the logs)
        log_dir (:class:`pathlib.Path`): Directory to store file-based logs in. If ``None``,
            get from :class:`~.mio.models.config.Config`. If ``False`` , disable file logging.
        level (:class:`.LOG_LEVELS`): Level to use for stdout logging. If ``None`` ,
            get from :class:`~.mio.models.config.Config`
        file_level (:class:`.LOG_LEVELS`): Level to use for file-based logging.
             If ``None`` , get from :class:`~.mio.models.config.Config`
        log_file_n (int): Number of rotating file logs to use.
            If ``None`` , get from :class:`~.mio.models.config.Config`
        log_file_size (int): Maximum size of logfiles before rotation.
            If ``None`` , get from :class:`~.mio.models.config.Config`

    Returns:
        :class:`logging.Logger`
    """
    config = get_config()
    if log_dir is None:
        log_dir = config.log_dir
    if level is None:
        level: LOG_LEVELS = (
            config.logs.level_stdout if config.logs.level_stdout is not None else config.logs.level
        )
    if file_level is None:
        file_level: LOG_LEVELS = (
            config.logs.level_file if config.logs.level_file is not None else config.logs.level
        )
    if log_file_n is None:
        log_file_n = config.logs.file_n
    if log_file_size is None:
        log_file_size = config.logs.file_size

    # set our logger to the minimum of the levels so that it always handles at least that severity
    # even if one or the other handlers might not.
    min_level = min([getattr(logging, level), getattr(logging, file_level)])

    if not name.startswith("mio"):
        name = "mio." + name

    _init_root(
        stdout_level=level,
        file_level=file_level,
        log_dir=log_dir,
        log_file_n=log_file_n,
        log_file_size=log_file_size,
    )

    logger = logging.getLogger(name)
    logger.setLevel(min_level)

    # if run from a forked process, need to add different handlers to not collide
    if mp.parent_process() is not None:
        logger.addHandler(
            _file_handler(
                name=f"{name}_{mp.current_process().pid}",
                file_level=file_level,
                log_dir=log_dir,
                log_file_n=log_file_n,
                log_file_size=log_file_size,
            )
        )
        logger.addHandler(_rich_handler(level))
        logger.propagate = False

    return logger


def _init_root(
    stdout_level: LOG_LEVELS,
    file_level: LOG_LEVELS,
    log_dir: Path,
    log_file_n: int = 5,
    log_file_size: int = 2**22,
) -> None:
    root_logger = logging.getLogger("mio")
    file_handlers = [
        handler for handler in root_logger.handlers if isinstance(handler, RotatingFileHandler)
    ]
    stream_handlers = [
        handler for handler in root_logger.handlers if isinstance(handler, RichHandler)
    ]

    if log_dir is not False and not file_handlers:
        root_logger.addHandler(
            _file_handler(
                "mio",
                file_level,
                log_dir,
                log_file_n,
                log_file_size,
            )
        )
    else:
        for file_handler in file_handlers:
            file_handler.setLevel(file_level)

    if not stream_handlers:
        root_logger.addHandler(_rich_handler(stdout_level))
    else:
        for stream_handler in stream_handlers:
            stream_handler.setLevel(stdout_level)

    # prevent propagation to the default root
    root_logger.propagate = False


def _file_handler(
    name: str,
    file_level: LOG_LEVELS,
    log_dir: Path,
    log_file_n: int = 5,
    log_file_size: int = 2**22,
) -> RotatingFileHandler:
    # See init_logger for arg docs

    filename = Path(log_dir) / ".".join([name, "log"])
    file_handler = RotatingFileHandler(
        str(filename), mode="a", maxBytes=log_file_size, backupCount=log_file_n
    )
    file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    return file_handler


def _rich_handler(level: LOG_LEVELS) -> RichHandler:
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_formatter = logging.Formatter(
        r"[bold green]\[%(name)s][/bold green] %(message)s",
        datefmt="[%y-%m-%dT%H:%M:%S]",
    )
    rich_handler.setFormatter(rich_formatter)
    rich_handler.setLevel(level)
    return rich_handler

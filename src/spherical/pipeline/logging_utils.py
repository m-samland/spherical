# src/spherical/pipeline/logging_utils.py
from __future__ import annotations

import functools
import importlib.metadata
import logging
import platform
import socket
import sys
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from multiprocessing import Queue
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

__all__ = ["get_pipeline_logger", "install_queue_listener", "remove_queue_listener"]

# One listener per interpreter
_listener: QueueListener | None = None


def get_pipeline_logger(name: str,
                        log_dir: Path,
                        verbose: bool = True,
                        max_mb: int = 10,
                        backups: int = 3,
                        json_log: bool = True) -> logging.Logger:
    """
    Return a configured logger unique to one reduction run.
    Safe to call many times in the same Python session.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    # Plain text formatter
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # JSON formatter for structured logging (optional)
    if json_log:
        json_fmt = JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s %(target)s %(band)s %(night)s %(step)s %(status)s')

    # ----------- destination in the *main* process -------------
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    # Regular rotating log (plain text)
    rf_handler = RotatingFileHandler(
        log_dir / f"reduction_{ts}.log",
        maxBytes=max_mb * 1_048_576,
        backupCount=backups,
    )
    rf_handler.setFormatter(fmt)

    # Optional JSON log file
    if json_log:
        json_handler = RotatingFileHandler(
            log_dir / f"reduction_{ts}.jsonlog",
            maxBytes=max_mb * 1_048_576,
            backupCount=backups,
        )
        json_handler.setFormatter(json_fmt)

    # Queue-based logging setup
    queue = Queue(-1)
    q_handler = QueueHandler(queue)
    logger.addHandler(q_handler)

    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # ----------- kick off listener once per interpreter --------
    global _listener
    if _listener is None:
        handler_list = [rf_handler]
        if json_log:
            handler_list.append(json_handler)
        _listener = QueueListener(
            queue,
            *handler_list,
            respect_handler_level=True
        )
        _listener.start()

    return logger


def install_queue_listener():  # for tests that run without pipeline
    """Start listener if it isn't running yet."""
    get_pipeline_logger("dummy", Path("/tmp"))


def remove_queue_listener():
    global _listener
    if _listener:
        _listener.stop()
        _listener = None

def optional_logger(fn):
    """Inject a default NullHandler logger when none is supplied."""
    @functools.wraps(fn)
    def wrapper(*args, logger: logging.Logger | None = None, **kw):
        if logger is None:
            logger = logging.getLogger(fn.__module__)
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
        return fn(*args, logger=logger, **kw)
    return wrapper

class PipelineLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that injects static pipeline context (target, band, night) into all log records.
    """
    def process(self, msg, kwargs):
        extra = self.extra.copy()
        if 'extra' in kwargs:
            extra.update(kwargs['extra'])
        kwargs['extra'] = extra
        return msg, kwargs

def get_pipeline_log_context(observation):
    def get_version(pkg):
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return 'unknown'
    return {
        "target": observation.target_name,
        "band": observation.obs_band,
        "night": observation.date,
        "spherical_version": get_version('spherical'),
        "charis_version": get_version('charis'),
        "trap_version": get_version('trap'),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
    }

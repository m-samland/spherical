# src/spherical/pipeline/logging_utils.py
from __future__ import annotations

import functools
import importlib.metadata
import logging
import platform
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from multiprocessing import Queue
from pathlib import Path

try:
    from pythonjsonlogger.json import JsonFormatter  # v3+
except ImportError:
    from pythonjsonlogger import jsonlogger

    JsonFormatter = jsonlogger.JsonFormatter  # v2

__all__ = [
    "bridge_library_logger",
    "get_pipeline_logger",
    "install_queue_listener",
    "remove_queue_listener",
]

# One listener per interpreter
_listener: QueueListener | None = None


def archive_old_pipeline_logs(log_dir: Path, log_files=("reduction.log", "reduction.jsonlog")):
    """
    Move any existing pipeline log files to an 'old_logs' backup folder within log_dir, renaming them with a timestamp.
    Ensures only the newest logs are present in log_dir at each run.
    Uses pathlib for filesystem operations.
    """
    old_logs_dir = log_dir / "old_logs"
    old_logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for log_file in log_files:
        log_path = log_dir / log_file
        if log_path.exists():
            new_name = f"{log_path.stem}_{ts}{log_path.suffix}"
            dest_path = old_logs_dir / new_name
            log_path.rename(dest_path)
    # Small wait to ensure filesystem operations complete
    time.sleep(0.3)


def get_pipeline_logger(name: str,
                        log_dir: Path,
                        verbose: bool = True,
                        max_mb: int = 10,
                        backups: int = 3,
                        json_log: bool = True,
                        log_prefix: str = "reduction") -> logging.Logger:
    """
    Return a configured logger unique to one reduction run.
    Safe to call many times in the same Python session.
    Moves any existing logs to a backup folder before creating new handlers.
    """
    global _listener

    logger = logging.getLogger(name)

    if logger.handlers:
        if _listener is not None:
            return logger  # already configured and still being drained
        # Logger names recur within one session whenever two observations share
        # a target/band/night. `remove_queue_listener()` stopped the listener
        # when the previous one finished, so the cached QueueHandler now feeds a
        # queue nobody drains: every record would be dropped, and the archiving
        # below would have moved the previous log away, leaving the folder with
        # no `{log_prefix}.jsonlog` for the monitoring scripts to find. Rebuild.
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    # Archive any old logs before creating new ones. Deliberately after the
    # reuse check — moving the file out from under a live handler would silently
    # redirect the running session's records into `old_logs/`.
    log_files = (f"{log_prefix}.log", f"{log_prefix}.jsonlog")
    archive_old_pipeline_logs(log_dir, log_files=log_files)

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
    # Regular rotating log (plain text)
    rf_handler = RotatingFileHandler(
        log_dir / f"{log_prefix}.log",
        maxBytes=max_mb * 1_048_576,
        backupCount=backups,
    )
    rf_handler.setFormatter(fmt)

    # Optional JSON log file
    if json_log:
        json_handler = RotatingFileHandler(
            log_dir / f"{log_prefix}.jsonlog",
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


@contextmanager
def bridge_library_logger(pipeline_logger, library_name="trap", level=logging.INFO):
    """Route a third-party library's log records into the per-target log files.

    `trap` logs its progress on its own logger tree, which the pipeline never
    configures: nothing in this package adds a root handler, so those records
    die at `logging.lastResort` (level WARNING). The practical effect is that a
    target that runs for hours writes a `trap_reduction.log` containing only the
    pipeline's own bookend messages — and, when it crashes, a traceback with no
    indication of what it was doing. `processing.verbose=True` does not help,
    because that flag only adds a stdout handler to the *pipeline* logger.

    Parameters
    ----------
    pipeline_logger : logging.Logger or logging.LoggerAdapter
        The per-target logger from `get_pipeline_logger`; its queue handler is
        borrowed so library records land in the same files.
    library_name : str
        Root name of the library's logger tree.
    level : int
        Level to force on that tree for the duration.

    Notes
    -----
    Library records carry none of the `target`/`band`/`night`/`step`/`status`
    fields, so `aggregate_reduction_status` ignores them — they enrich the
    human-readable log without polluting status aggregation.
    """
    logger = getattr(pipeline_logger, "logger", pipeline_logger)
    queue_handlers = [h for h in logger.handlers if isinstance(h, QueueHandler)]

    library_logger = logging.getLogger(library_name)
    previous_level = library_logger.level
    added = []
    try:
        for handler in queue_handlers:
            if handler not in library_logger.handlers:
                library_logger.addHandler(handler)
                added.append(handler)
        # NOTSET would inherit root's WARNING and drop exactly the progress
        # messages this bridge exists to capture.
        library_logger.setLevel(level)
        yield library_logger
    finally:
        for handler in added:
            library_logger.removeHandler(handler)
        library_logger.setLevel(previous_level)


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

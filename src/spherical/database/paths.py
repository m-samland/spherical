"""Resolution of the directory holding the database tables.

One directory is shared by every entry point: the CLIs write it, the reduction
templates read the observation and file tables from it, and ``plot_trap_mosaics``
annotates its figures from it. ``$SPHERICAL_DATABASE_DIR`` lets a user name that
directory once for the whole install instead of repeating it in each command.

Kept free of non-stdlib imports so the base install (no ``pipeline`` extra) can
use it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

ENV_DATABASE_DIR = "SPHERICAL_DATABASE_DIR"

__all__ = ["ENV_DATABASE_DIR", "database_dir_from_env", "resolve_database_dir"]

PathLike = Union[str, os.PathLike]


def database_dir_from_env() -> Optional[Path]:
    """Return ``$SPHERICAL_DATABASE_DIR`` as a path, or ``None`` if unset/empty."""
    value = os.environ.get(ENV_DATABASE_DIR, "").strip()
    return Path(value).expanduser() if value else None


def resolve_database_dir(
    explicit: Optional[PathLike] = None,
    default: Optional[PathLike] = None,
) -> Optional[Path]:
    """Resolve the database directory: *explicit* wins, then the environment, then *default*.

    Parameters
    ----------
    explicit : str or path-like, optional
        A directory the caller was given directly (a CLI flag, a function
        argument). Always wins when not ``None``.
    default : str or path-like, optional
        Fallback used when neither *explicit* nor the environment variable is
        set. ``None`` means "no directory", which callers treat as either an
        error or a degraded mode.

    Returns
    -------
    pathlib.Path or None
        The resolved directory, with ``~`` expanded, or ``None`` when nothing
        resolved.
    """
    if explicit is not None:
        return Path(explicit).expanduser()

    from_env = database_dir_from_env()
    if from_env is not None:
        logger.info("Using database directory from $%s: %s", ENV_DATABASE_DIR, from_env)
        return from_env

    return Path(default).expanduser() if default is not None else None

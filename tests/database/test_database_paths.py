"""Resolution order for the database-table directory."""
from pathlib import Path

import pytest

from spherical.database.paths import (
    ENV_DATABASE_DIR,
    database_dir_from_env,
    resolve_database_dir,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """A developer's own $SPHERICAL_DATABASE_DIR must not leak into these tests."""
    monkeypatch.delenv(ENV_DATABASE_DIR, raising=False)


def test_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_DATABASE_DIR, "/from/env")
    assert resolve_database_dir("/from/cli", default="/from/default") == Path("/from/cli")


def test_env_wins_over_default(monkeypatch):
    monkeypatch.setenv(ENV_DATABASE_DIR, "/from/env")
    assert resolve_database_dir(None, default="/from/default") == Path("/from/env")


def test_default_used_when_env_unset():
    assert resolve_database_dir(None, default="/from/default") == Path("/from/default")


def test_none_when_nothing_set():
    assert resolve_database_dir(None) is None
    assert database_dir_from_env() is None


def test_empty_env_is_treated_as_unset(monkeypatch):
    """An exported-but-empty variable must not resolve to the current directory."""
    monkeypatch.setenv(ENV_DATABASE_DIR, "   ")
    assert database_dir_from_env() is None
    assert resolve_database_dir(None, default="/from/default") == Path("/from/default")


def test_tilde_is_expanded(monkeypatch):
    monkeypatch.setenv(ENV_DATABASE_DIR, "~/data/sphere/database")
    resolved = resolve_database_dir(None)
    assert resolved == Path.home() / "data/sphere/database"
    assert "~" not in str(resolved)


def test_explicit_path_object_accepted(monkeypatch):
    monkeypatch.setenv(ENV_DATABASE_DIR, "/from/env")
    assert resolve_database_dir(Path("/from/cli")) == Path("/from/cli")

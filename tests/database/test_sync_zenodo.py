from pathlib import Path
from unittest.mock import patch

import pytest

from spherical.database.paths import ENV_DATABASE_DIR
from spherical.scripts import sync_zenodo_tables as sz


@pytest.fixture(autouse=True)
def _clear_database_dir_env(monkeypatch):
    """A developer's own $SPHERICAL_DATABASE_DIR must not leak into these tests."""
    monkeypatch.delenv(ENV_DATABASE_DIR, raising=False)


def test_wanted_filenames_all_includes_polarimetry():
    names = sz._wanted_filenames("all", include_polarimetry=True)
    assert "table_of_observations_irdis_polarimetry.fits" in names
    assert "table_of_files_ifs.csv" in names


def test_wanted_filenames_ifs_only():
    names = sz._wanted_filenames("ifs", include_polarimetry=False)
    assert all("irdis" not in n for n in names)


def test_checksum_missing_file_is_false(tmp_path):
    assert sz._checksum_matches(tmp_path / "nope.csv", "md5:abc") is False


def test_checksum_non_md5_is_unverifiable(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("hello")
    # sha256-style checksum: cannot verify, must not claim a match nor a mismatch.
    assert sz._checksum_matches(f, "sha256:deadbeef") is False
    assert sz.checksum_verifiable("sha256:deadbeef") is False


def test_checksum_md5_matches(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"hello")
    import hashlib
    md5 = hashlib.md5(b"hello").hexdigest()
    assert sz._checksum_matches(f, f"md5:{md5}") is True
    assert sz.checksum_verifiable(f"md5:{md5}") is True


def test_dest_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DATABASE_DIR, str(tmp_path))
    with patch.object(sz, "sync_tables", return_value=0) as sync:
        assert sz.main([]) == 0
    assert sync.call_args.kwargs["dest"] == Path(str(tmp_path))


def test_explicit_dest_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DATABASE_DIR, str(tmp_path / "from_env"))
    explicit = tmp_path / "from_cli"
    with patch.object(sz, "sync_tables", return_value=0) as sync:
        sz.main(["--dest", str(explicit)])
    assert sync.call_args.kwargs["dest"] == explicit


def test_missing_dest_and_env_errors():
    with pytest.raises(SystemExit) as excinfo:
        sz.main([])
    assert excinfo.value.code == 2


def test_dry_run_still_works_without_dest_or_env():
    with patch.object(sz, "sync_tables", return_value=0) as sync:
        assert sz.main(["--dry-run"]) == 0
    assert sync.call_args.kwargs["dest"] == Path(".")

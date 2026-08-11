import json
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


def test_sam_tables_are_opt_in():
    without = sz._wanted_filenames("all", include_polarimetry=False)
    assert not any("_sam" in n for n in without)

    with_sam = sz._wanted_filenames("all", include_polarimetry=False, include_sam=True)
    assert "table_of_observations_ifs_sam.fits" in with_sam
    assert "table_of_observations_irdis_sam.fits" in with_sam


def test_target_tables_are_optional_not_required():
    """Requiring them would break syncing against the v2.0.0 record, which has none."""
    required = sz._wanted_filenames("all", include_polarimetry=True)
    assert not any(n.startswith("table_of_targets_") for n in required)

    optional = sz._optional_filenames("all", include_polarimetry=True)
    assert "table_of_targets_ifs.fits" in optional
    assert "table_of_targets_irdis.fits" in optional
    assert "table_of_targets_irdis_polarimetry.fits" in optional
    assert sz.PROVENANCE_NAME in optional


def test_optional_filenames_follow_instrument_and_sam_selection():
    optional = sz._optional_filenames("ifs", include_polarimetry=False, include_sam=True)
    assert "table_of_targets_ifs_sam.fits" in optional
    assert not any("irdis" in n for n in optional if n.startswith("table_of_targets_"))


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


V2_FILES = [
    "table_of_files_ifs.csv",
    "table_of_files_irdis.csv",
    "table_of_observations_ifs.fits",
    "table_of_observations_irdis.fits",
    "table_of_observations_irdis_polarimetry.fits",
]

V3_FILES = V2_FILES + [
    "table_of_targets_ifs.fits",
    "table_of_targets_irdis.fits",
    "table_of_targets_irdis_polarimetry.fits",
    sz.PROVENANCE_NAME,
]


def _fake_record(filenames):
    return {
        "metadata": {"version": "v3.0.0"},
        "files": [{"key": name, "links": {"self": f"https://example.invalid/{name}"}} for name in filenames],
    }


def _run_sync(monkeypatch, dest, filenames, **kwargs):
    """Drive sync_tables against a fabricated record, writing plausible file content."""
    monkeypatch.setattr(sz, "_resolve_zenodo_record_id", lambda *a, **k: "999")
    monkeypatch.setattr(sz, "_load_record", lambda *a, **k: _fake_record(filenames))

    downloaded = []

    def fake_download(url, destination, timeout=None):
        downloaded.append(destination.name)
        if destination.name.endswith(".csv"):
            destination.write_text("NIGHT_START\n2014-05-14\n2026-08-09\n")
        elif destination.name == sz.PROVENANCE_NAME:
            destination.write_text(json.dumps({"tables": {"ifs": {"source": "eso-extend", "mode": "ifs"}}}))
        else:
            destination.write_bytes(b"\x00")

    monkeypatch.setattr(sz, "_download_file", fake_download)

    params = dict(
        doi_or_record="10.5281/zenodo.15147730",
        dest=dest,
        instrument="all",
        include_polarimetry=True,
        timeout=5,
        force=False,
    )
    params.update(kwargs)
    assert sz.sync_tables(**params) == 0
    return downloaded


def test_sync_tolerates_record_without_targets_or_provenance(monkeypatch, tmp_path):
    """The published v2.0.0 record has neither; the sync must still succeed."""
    downloaded = _run_sync(monkeypatch, tmp_path, V2_FILES)
    assert sorted(downloaded) == sorted(V2_FILES)


def test_sync_downloads_targets_and_provenance_when_offered(monkeypatch, tmp_path):
    downloaded = _run_sync(monkeypatch, tmp_path, V3_FILES)
    assert "table_of_targets_ifs.fits" in downloaded
    assert sz.PROVENANCE_NAME in downloaded


def test_sync_keeps_downloaded_provenance_intact(monkeypatch, tmp_path):
    """Reconstruction would replace the ifs/irdis entries with source=zenodo stubs."""
    _run_sync(monkeypatch, tmp_path, V3_FILES)

    written = json.loads((tmp_path / sz.PROVENANCE_NAME).read_text())
    assert written["tables"]["ifs"]["source"] == "eso-extend"


def test_sync_reconstructs_provenance_for_legacy_record(monkeypatch, tmp_path):
    _run_sync(monkeypatch, tmp_path, V2_FILES)

    written = json.loads((tmp_path / sz.PROVENANCE_NAME).read_text())
    assert written["tables"]["ifs"]["source"] == "zenodo"
    assert written["tables"]["ifs"]["eso_coverage_end"] == "2026-08-09"

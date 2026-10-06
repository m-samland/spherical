"""Notebook helpers: private paths, short logs, and a clear stop when a run is missing."""

import json
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "tutorials"))
import tutorial_helpers as th  # noqa: E402


def test_tidy_path_replaces_home():
    assert th.tidy_path(Path.home() / "data" / "x.fits") == "~/data/x.fits"
    assert th.tidy_path("/opt/data/x.fits") == "/opt/data/x.fits"


@pytest.mark.parametrize("path", ["/home/someone/data/x.fits", "/u/someone/data/x.fits",
                                  "/Users/someone/data/x.fits"])
def test_tidy_path_hides_other_machines_homes(path):
    # The IFS logs are written on a server and printed on another machine.
    assert th.tidy_path(f"INFO reading {path}") == "INFO reading ~/data/x.fits"


def test_tutorial_dir_reads_environment_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    assert th.tutorial_dir() == tmp_path


def test_show_log_excerpt_truncates_long_logs(tmp_path, capsys):
    log = tmp_path / "reduction.log"
    log.write_text("".join(f"line {i} {Path.home()}\n" for i in range(100)))
    th.show_log_excerpt(log)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 31
    assert lines[15] == "... 70 lines omitted ..."
    assert lines[0] == "line 0 ~" and lines[-1] == "line 99 ~"


def test_show_log_excerpt_prints_short_logs_whole(tmp_path, capsys):
    log = tmp_path / "reduction.log"
    log.write_text("".join(f"line {i}\n" for i in range(10)))
    th.show_log_excerpt(log)
    assert len(capsys.readouterr().out.splitlines()) == 10


def _run(tmp_path, monkeypatch, files=("converted/a.fits",), source_hash="abc"):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    run = tmp_path / "51eri_irdis"
    for name in files:
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("x")
    (run / "run_provenance.json").write_text(json.dumps({"source_hash": source_hash}))
    return run


def test_require_run_returns_complete_run(tmp_path, monkeypatch):
    run = _run(tmp_path, monkeypatch)
    assert th.require_run("51eri_irdis", ["converted/a.fits"]) == run


def test_require_run_names_script_and_missing_files(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match=r"runs/51eri_irdis\.py") as error:
        th.require_run("51eri_irdis", ["converted/a.fits", "trap/b.csv"])
    assert "trap/b.csv" in str(error.value) and "converted/a.fits" not in str(error.value)


def test_require_run_without_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="run_provenance.json"):
        th.require_run("51eri_irdis", [])


def test_require_run_warns_on_source_change(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, source_hash="abc")
    with pytest.warns(UserWarning, match="changed since the run"):
        th.require_run("51eri_irdis", [], source_hash="def")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        th.require_run("51eri_irdis", [], source_hash="abc")

"""The tutorial stamp records versions and the machine, never who or where."""

import datetime
import getpass
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "tools"))
import stamp_notebook  # noqa: E402

TODAY = datetime.date(2026, 10, 20)


def test_collect_provenance_keys_and_privacy():
    prov = stamp_notebook.collect_provenance(ncpu=4, today=TODAY)
    assert tuple(prov) == stamp_notebook.PROVENANCE_KEYS
    assert prov["date"] == "2026-10-20" and prov["ncpu"] == 4
    assert isinstance(prov["cores"], int) and prov["cores"] > 0
    text = json.dumps(prov)
    assert socket.gethostname() not in text
    assert getpass.getuser() not in text


def test_spherical_version_comes_from_git(monkeypatch):
    monkeypatch.setattr(stamp_notebook, "_git_describe", lambda: "v3.2.0rc1-3-gabc1234")
    assert stamp_notebook.collect_provenance(None, TODAY)["spherical"] == "v3.2.0rc1-3-gabc1234"


def test_write_run_provenance_adds_extra(tmp_path):
    path = tmp_path / "51eri_irdis" / "run_provenance.json"
    stamp_notebook.write_run_provenance(path, ncpu=8, extra={"label": "51eri_irdis"})
    data = json.loads(path.read_text())
    assert data["ncpu"] == 8 and data["label"] == "51eri_irdis"


def test_resume_keeps_original_and_rejects_new_version(tmp_path, monkeypatch):
    path = tmp_path / "x" / "run_provenance.json"
    monkeypatch.setattr(stamp_notebook, "_git_describe", lambda: "v3.2.0")
    stamp_notebook.write_run_provenance(path, ncpu=8)
    stamp_notebook.write_run_provenance(path, ncpu=8)
    data = json.loads(path.read_text())
    assert data["spherical"] == "v3.2.0" and len(data["resumes"]) == 1
    monkeypatch.setattr(stamp_notebook, "_git_describe", lambda: "v3.2.1")
    with pytest.raises(ValueError, match="v3.2.1"):
        stamp_notebook.write_run_provenance(path, ncpu=8)


def _notebook(tmp_path, name="t"):
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# T"]}],
          "metadata": {"kernelspec": {"name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / f"{name}.ipynb"
    path.write_text(json.dumps(nb))
    return path, nb


def test_stamp_writes_run_and_rendered(tmp_path):
    path, nb = _notebook(tmp_path)
    stamp_notebook.stamp(path, {"spherical": "3.2.0"}, rendered=TODAY)
    out = json.loads(path.read_text())
    assert out["metadata"]["spherical"] == {"run": {"spherical": "3.2.0"}, "rendered": "2026-10-20"}
    assert out["metadata"]["kernelspec"] == {"name": "python3"} and out["cells"] == nb["cells"]
    assert path.read_text().endswith("\n")


def test_stamp_from_run_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path / "tut"))
    stamp_notebook.write_run_provenance(tmp_path / "tut" / "51eri_irdis" / "run_provenance.json", ncpu=6)
    path, _ = _notebook(tmp_path, "51eri_irdis")
    stamp_notebook.main([str(path)])
    assert json.loads(path.read_text())["metadata"]["spherical"]["run"]["ncpu"] == 6


def test_stamp_without_run_uses_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path / "empty"))
    path, _ = _notebook(tmp_path, "exploring_the_database")
    stamp_notebook.main([str(path)])
    assert json.loads(path.read_text())["metadata"]["spherical"]["run"]["ncpu"] is None


def test_main_drops_widget_outputs(tmp_path, monkeypatch):
    # tqdm.notebook ignores TQDM_DISABLE, so progress bars arrive as widget outputs.
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path / "empty"))
    widget = {"output_type": "display_data", "metadata": {},
              "data": {"application/vnd.jupyter.widget-view+json": {"model_id": "x"}, "text/plain": "0%|"}}
    text = {"output_type": "stream", "name": "stdout", "text": "kept\n"}
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": [], "execution_count": None,
                     "outputs": [widget, text]}],
          "metadata": {"widgets": {"state": {}}}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "t.ipynb"
    path.write_text(json.dumps(nb))
    stamp_notebook.main([str(path)])
    out = json.loads(path.read_text())
    assert out["cells"][0]["outputs"] == [text]
    assert "widgets" not in out["metadata"]

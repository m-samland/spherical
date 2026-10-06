"""The reduction templates build their configuration in functions, and the tutorial run
scripts reuse exactly those settings."""

import dataclasses
import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trap")
pytest.importorskip("charis")

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "docs" / "tutorials" / "runs"))

TEMPLATES = {"ifs": "ifs_reduction_template", "irdis": "irdis_reduction_template"}
RUN_SCRIPTS = {"ifs": "51eri_ifs", "irdis": "51eri_irdis"}


def normalise(value):
    """Plain data for comparison; TRAP configs hold numpy arrays, whose == is ambiguous."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: normalise(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, (Path, range)):
        return str(value)
    return value


def fresh_import(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_import_has_no_side_effects(instrument, monkeypatch, capsys):
    import astropy.table

    def refuse(*args, **kwargs):
        raise AssertionError("a template read a table at import")

    monkeypatch.setattr(astropy.table.Table, "read", refuse)
    fresh_import(TEMPLATES[instrument])
    assert capsys.readouterr().out == ""


def test_ifs_trap_settings():
    template = fresh_import("ifs_reduction_template")
    trap_config = template.build_trap_config(template.build_config())
    assert trap_config.reduction.search_region_outer_bound == 65
    assert trap_config.detection.search_radius == 15
    assert trap_config.detection.candidate_threshold == 4.75
    assert trap_config.detection.detection_threshold == 5.0
    assert list(trap_config.processing.temporal_components_fraction) == [0.15]


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_build_config_sets_cpu_and_paths(instrument, tmp_path):
    template = fresh_import(TEMPLATES[instrument])
    config = template.build_config(ncpu=3, base_path=tmp_path)
    assert config.resources.ncpu == 3
    assert Path(config.directories.reduction_directory) == tmp_path / "reduction"
    assert Path(config.directories.raw_directory) == tmp_path / "data"


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_run_script_uses_template_trap_settings(instrument, tmp_path, monkeypatch):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    template = fresh_import(TEMPLATES[instrument])
    run = fresh_import(RUN_SCRIPTS[instrument])
    config, trap_config = run.build(ncpu=6, label=RUN_SCRIPTS[instrument])
    expected_config = template.build_config(ncpu=6)
    assert normalise(trap_config) == normalise(template.build_trap_config(expected_config))
    for field in dataclasses.fields(config):
        if field.name != "directories":
            assert normalise(getattr(config, field.name)) == normalise(getattr(expected_config, field.name)), field.name
    assert Path(config.directories.raw_directory) == tmp_path / "data"
    assert Path(config.directories.reduction_directory) == tmp_path / RUN_SCRIPTS[instrument] / "reduction"


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_source_hash_changes_with_source(instrument, monkeypatch):
    run = fresh_import(RUN_SCRIPTS[instrument])
    first = run.source_hash()
    assert run.source_hash() == first
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource",
                        lambda obj: real(obj) + ("# edited\n" if obj.__name__ == "build_trap_config" else ""))
    assert run.source_hash() != first


def test_run_script_refuses_non_empty_reduction(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    run = fresh_import("51eri_irdis")
    (tmp_path / "51eri_irdis" / "reduction" / "IRDIS").mkdir(parents=True)

    def no_selection(*args, **kwargs):
        raise AssertionError("selected observations before refusing")

    monkeypatch.setattr(run.template, "select_observations", no_selection)
    assert run.main(["--ncpu", "2"]) == 1
    assert not (tmp_path / "51eri_irdis" / "run_provenance.json").exists()


def test_run_script_writes_provenance_before_reducing(tmp_path, monkeypatch):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    run = fresh_import("51eri_irdis")
    calls = []
    monkeypatch.setattr(run.template, "select_observations",
                        lambda targets, **criteria: (["row"], ["observation"]))
    monkeypatch.setattr(run.runner, "execute_targets",
                        lambda **kwargs: calls.append(("reduce", (tmp_path / "51eri_irdis" / "run_provenance.json").exists())))
    monkeypatch.setattr(run.runner, "run_trap_on_observations", lambda **kwargs: calls.append(("trap", True)))
    assert run.main(["--ncpu", "2", "--species-dir", str(tmp_path / "species")]) == 0
    assert calls == [("reduce", True), ("trap", True)]
    provenance = json.loads((tmp_path / "51eri_irdis" / "run_provenance.json").read_text())
    assert provenance["obs_id"] == 200363269 and provenance["source_hash"] == run.source_hash()


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_run_script_starts_from_any_directory(instrument, tmp_path):
    # Run as a script, nothing but the script itself puts examples/ on sys.path.
    import subprocess
    script = ROOT / "docs" / "tutorials" / "runs" / f"{RUN_SCRIPTS[instrument]}.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr

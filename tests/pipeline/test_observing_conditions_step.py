"""The observing_conditions pipeline step: registry entry, wrapper, plot and IRDIS driver wiring.

The IFS driver test lives in test_ifs_cleanup_resume.py, which skips without charis.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from astropy.table import Table
from test_conditions_run import Network, _frames, _sparta_hdus
from test_conditions_sparta import T0

import spherical.conditions
from spherical.conditions import ConditionsSettings
from spherical.pipeline import step_registry
from spherical.pipeline.pipeline_config import PreprocConfig, defaultIRDISReduction
from spherical.pipeline.step_registry import force_needs_cubes
from spherical.pipeline.steps import observing_conditions as step
from spherical.pipeline.steps.download_data import raw_science_dir

LOGGER = logging.getLogger("test_observing_conditions_step")


def test_registry_places_step_after_frames_info():
    for registry in (step_registry.STEP_REGISTRY, step_registry.IRDIS_STEP_REGISTRY):
        order = list(registry)
        assert order[order.index("compute_frames_info") + 1] == "observing_conditions"
        assert registry["observing_conditions"].leaf
    outputs = step_registry.expected_outputs("observing_conditions", step_registry.StepDirs(converted_dir=Path("/c")))
    assert outputs == [Path("/c/conditions") / spherical.conditions.COVERAGE_FILE]


def test_force_needs_cubes():
    assert force_needs_cubes({"observing_conditions"}) is False
    assert force_needs_cubes({"observing_conditions", "find_centers"}) is True
    assert force_needs_cubes(True) is True
    assert force_needs_cubes(set()) is True
    assert force_needs_cubes(False) is True


def test_raw_science_dir_matches_the_download_layout(tmp_path):
    assert raw_science_dir(tmp_path, "irdis", "*  51 Eri", "DB_K12", "2015-09-24") == (
        tmp_path / "IRDIS" / "science" / "*_51_Eri" / "DB_K12" / "2015-09-24"
    )


def test_step_wrapper_never_raises(tmp_path, monkeypatch, caplog):
    _frames(T0).to_csv(tmp_path / "frames_info_coro.csv", index=False)

    def boom(*args, **kwargs):
        raise RuntimeError("archive exploded")

    monkeypatch.setattr(step, "run_conditions", boom)
    with caplog.at_level(logging.ERROR):
        result = step.run_observing_conditions(tmp_path, tmp_path / "sparta", ConditionsSettings(), PreprocConfig(), LOGGER)
    assert result is None
    assert "archive exploded" in caplog.text
    assert [r.status for r in caplog.records if getattr(r, "step", None) == "observing_conditions"][-1] == "failed"


def test_plot_written(tmp_path, monkeypatch):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    converted = tmp_path / "converted"
    converted.mkdir()
    _frames(T0 + 20).to_csv(converted / "frames_info_coro.csv", index=False)
    _frames(T0 + 5, count=2, dit=2.0).to_csv(converted / "frames_info_flux.csv", index=False)

    step.run_observing_conditions(converted, tmp_path / "sparta", ConditionsSettings(), PreprocConfig(), LOGGER)

    out = converted / "conditions"
    assert (out / spherical.conditions.COVERAGE_FILE).exists()
    assert (out / "matched_coro.csv").exists()
    assert (out / "conditions.png").stat().st_size > 0


def test_plot_with_no_sparta_data(tmp_path, monkeypatch, caplog):
    Network(tmp_path, monkeypatch, ambient_fail={"mass", "dimm", "meteo", "lhatpro"})
    converted = tmp_path / "converted"
    converted.mkdir()
    _frames(T0 + 20).to_csv(converted / "frames_info_center.csv", index=False)

    with caplog.at_level(logging.INFO):
        step.run_observing_conditions(converted, tmp_path / "sparta", ConditionsSettings(), PreprocConfig(), LOGGER)

    # An archive outage leaves a healthy reduction; reduction_status must not list it as failed.
    statuses = [r.status for r in caplog.records if getattr(r, "step", None) == "observing_conditions"]
    assert statuses[-1] == "warning"
    assert "failed" not in statuses

    assert (converted / "conditions" / "conditions.png").exists()
    assert not (converted / "conditions" / spherical.conditions.COVERAGE_FILE).exists()


def test_registry_coverage_name_matches_conditions_package():
    assert step_registry.CONDITIONS_COVERAGE_FILE == spherical.conditions.COVERAGE_FILE


def _irdis_observation():
    frame = Table({"DP.ID": ["SPHER.2024-01-01T00:00:00.000"]})
    return SimpleNamespace(
        observation=Table({
            "INSTRUMENT": ["irdis"], "MAIN_ID": ["TEST TARGET"], "FILTER": ["DB_H23"],
            "NIGHT_START": ["2024-01-01"], "WAFFLE_MODE": [False],
        }),
        frames={"CORO": frame, "CENTER": frame, "FLUX": frame, "FLAT": frame, "BG_SCIENCE": frame},
        filter="DB_H23",
    )


def _irdis_converted(tmp_path):
    converted = tmp_path / "reduction/IRDIS/observation/TEST_TARGET/DB_H23/2024-01-01/converted"
    converted.mkdir(parents=True)
    for frame_type in ("coro", "center", "flux"):
        pd.DataFrame({"MJD START": [60310.0], "MJD END": [60310.001]}).to_csv(
            converted / f"frames_info_{frame_type}.csv", index=False
        )
    return converted


def _irdis_config(tmp_path, force):
    config = defaultIRDISReduction()
    config.directories.raw_directory = tmp_path / "raw"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.enable_all_irdis_steps()
    config.steps.force = force
    return config


def test_forcing_conditions_on_cleaned_irdis_observation_does_not_rebuild(tmp_path):
    from spherical.pipeline import irdis_reduction as ir

    converted = _irdis_converted(tmp_path)
    marker = converted / ".cubes.cleaned"
    marker.touch()
    config = _irdis_config(tmp_path, {"observing_conditions"})

    with patch.object(ir, "download_data_for_observation") as download, patch.object(
        ir, "run_irdis_calibration") as calib, patch.object(ir, "run_irdis_preprocess") as pre, patch.object(
        ir, "run_frame_info_computation") as frames, patch.object(ir, "run_observing_conditions") as conditions:
        ir.execute_irdis_target(observation=_irdis_observation(), config=config)

    download.assert_not_called()
    calib.assert_not_called()
    pre.assert_not_called()
    frames.assert_not_called()
    conditions.assert_called_once()
    args = conditions.call_args.args
    assert args[0] == converted
    assert args[1] == tmp_path / "raw/IRDIS/science/TEST_TARGET/DB_H23/2024-01-01/SPARTA"
    assert args[2] is config.conditions and args[3] is config.preprocessing
    assert marker.exists()


def test_cleaned_irdis_observation_without_frame_tables_skips_conditions(tmp_path):
    from spherical.pipeline import irdis_reduction as ir

    converted = _irdis_converted(tmp_path)
    for path in converted.glob("frames_info_*.csv"):
        path.unlink()
    (converted / ".cubes.cleaned").touch()
    config = _irdis_config(tmp_path, {"observing_conditions"})

    with patch.object(ir, "download_data_for_observation") as download, patch.object(
        ir, "run_observing_conditions") as conditions:
        ir.execute_irdis_target(observation=_irdis_observation(), config=config)

    download.assert_not_called()
    conditions.assert_not_called()
    log = (converted.parent / "reduction.log").read_text()
    assert "no frames_info_*.csv" in log


def test_cleaned_irdis_observation_with_conditions_disabled_says_so(tmp_path):
    from spherical.pipeline import irdis_reduction as ir

    converted = _irdis_converted(tmp_path)
    (converted / ".cubes.cleaned").touch()
    config = _irdis_config(tmp_path, {"observing_conditions"})
    config.steps.observing_conditions = False

    with patch.object(ir, "run_observing_conditions") as conditions:
        ir.execute_irdis_target(observation=_irdis_observation(), config=config)

    conditions.assert_not_called()
    assert "observing_conditions is disabled" in (converted.parent / "reduction.log").read_text()

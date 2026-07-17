"""Tests for the IRDIS Phase 1 orchestrator (download-only skeleton)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from astropy.table import Table


def _make_irdis_observation(tmp_path):
    """Build a minimal IRDISObservation-like stand-in for orchestrator tests.

    Uses SimpleNamespace to avoid needing real IRDISObservation construction
    (which requires large table fixtures). The orchestrator only touches the
    attributes accessed below.
    """
    obs_row = Table(
        {
            "INSTRUMENT": ["irdis"],
            "MAIN_ID": ["TEST_TARGET"],
            "FILTER": ["DB_H23"],
            "NIGHT_START": ["2024-01-01"],
        }
    )
    frames = {
        "CORO": Table({"DP.ID": ["SPHER.2024-01-01T00:00:00.000"]}),
        "CENTER": Table({"DP.ID": ["SPHER.2024-01-01T00:01:00.000"]}),
        "FLUX": Table({"DP.ID": ["SPHER.2024-01-01T00:02:00.000"]}),
        "FLAT": Table({"DP.ID": ["SPHER.2024-01-01T00:03:00.000"]}),
        "BG_SCIENCE": Table({"DP.ID": ["SPHER.2024-01-01T00:04:00.000"]}),
    }
    return SimpleNamespace(
        observation=obs_row,
        frames=frames,
        filter="DB_H23",
        target_name=None,
        obs_band=None,
        date=None,
    )


def test_execute_irdis_target_download_only_calls_download(tmp_path):
    from spherical.pipeline.irdis_reduction import execute_irdis_target
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    observation = _make_irdis_observation(tmp_path)

    config = defaultIRDISReduction()
    config.directories.base_path = tmp_path
    config.directories.raw_directory = tmp_path / "data"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.disable_all_ifs_steps()
    config.steps = config.steps.merge(download_data=True)

    with patch(
        "spherical.pipeline.irdis_reduction.download_data_for_observation"
    ) as mocked:
        execute_irdis_target(observation=observation, config=config)

    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert kwargs["raw_directory"] == str(tmp_path / "data")
    passed_obs = kwargs["observation"]
    assert passed_obs.observation["MAIN_ID"][0] == "TEST_TARGET"
    assert passed_obs.observation["FILTER"][0] == "DB_H23"
    assert passed_obs.target_name == "TEST_TARGET"
    assert passed_obs.obs_band == "DB_H23"
    assert passed_obs.date == "2024-01-01"


def test_output_directory_path_no_method_segment(tmp_path):
    from spherical.pipeline.irdis_reduction import output_directory_path

    observation = _make_irdis_observation(tmp_path)
    path = output_directory_path(str(tmp_path / "reduction"), observation)
    assert "IRDIS/observation" in path
    assert "TEST_TARGET" in path
    assert "DB_H23" in path
    assert "2024-01-01" in path
    assert "optext" not in path
    assert "apphot" not in path

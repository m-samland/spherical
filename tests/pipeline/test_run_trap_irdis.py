"""Tests for the IRDIS branch of run_trap.py (Phase 6).

The heavy TRAP call (``run_complete_reduction``, ``DetectionAnalysis``) is out
of scope for CI — the tests here cover the small pure helpers only:

* instrument detection,
* coronagraph-transmission dispatch,
* result-folder path helper.

End-to-end IRDIS TRAP validation is Task 4 on real reference data.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np


class TestInstrumentOf:
    def test_ifs(self):
        from spherical.pipeline.run_trap import _instrument_of

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IFS"]}
        assert _instrument_of(observation) == "IFS"

    def test_irdis(self):
        from spherical.pipeline.run_trap import _instrument_of

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}
        assert _instrument_of(observation) == "IRDIS"

    def test_lowercase_input_uppercased(self):
        from spherical.pipeline.run_trap import _instrument_of

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["irdis"]}
        assert _instrument_of(observation) == "IRDIS"


class TestResultFolderFor:
    def test_ifs_layout(self):
        from spherical.pipeline.run_trap import _result_folder_for

        result = _result_folder_for("IFS", "/tmp/red", "bet_Pic/OBS_YJ/2015-01-01")
        assert result == "/tmp/red/IFS/trap/bet_Pic/OBS_YJ/2015-01-01"

    def test_irdis_layout_has_no_method_segment(self):
        from spherical.pipeline.run_trap import _result_folder_for

        result = _result_folder_for("IRDIS", "/tmp/red", "bet_Pic/DB_K12/2014-12-07")
        assert result == "/tmp/red/IRDIS/trap/bet_Pic/DB_K12/2014-12-07"


class TestCoronagraphResolution:
    def test_irdis_returns_irdis_table(self):
        from spherical.pipeline import run_trap

        reduction_config = MagicMock()
        reduction_config.apply_coronagraph_transmission = True

        trap_reduction_config = MagicMock()
        trap_reduction_config.coronagraph_transmission = None

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}

        table = run_trap._resolve_coronagraph_transmission(
            reduction_config, trap_reduction_config, observation
        )
        assert table is not None
        assert table.ndim == 2 and table.shape[1] == 2

    def test_ifs_returns_ifs_table(self):
        from spherical.pipeline import run_trap

        reduction_config = MagicMock()
        reduction_config.apply_coronagraph_transmission = True

        trap_reduction_config = MagicMock()
        trap_reduction_config.coronagraph_transmission = None

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IFS"]}

        table = run_trap._resolve_coronagraph_transmission(
            reduction_config, trap_reduction_config, observation
        )
        assert table is not None
        assert table.ndim == 2 and table.shape[1] == 2

    def test_explicit_table_wins(self):
        from spherical.pipeline import run_trap

        reduction_config = MagicMock()
        reduction_config.apply_coronagraph_transmission = True

        explicit = np.array([[0.0, 0.5], [100.0, 1.0]])
        trap_reduction_config = MagicMock()
        trap_reduction_config.coronagraph_transmission = explicit

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}

        assert run_trap._resolve_coronagraph_transmission(
            reduction_config, trap_reduction_config, observation
        ) is None

    def test_disabled_toggle_returns_none(self):
        from spherical.pipeline import run_trap

        reduction_config = MagicMock()
        reduction_config.apply_coronagraph_transmission = False

        trap_reduction_config = MagicMock()
        trap_reduction_config.coronagraph_transmission = None

        observation = MagicMock()
        observation.observation = {"INSTRUMENT": ["IRDIS"]}

        assert run_trap._resolve_coronagraph_transmission(
            reduction_config, trap_reduction_config, observation
        ) is None


class TestBatchErrorIsolation:
    """One target dying must not take the rest of the batch with it.

    `run_trap_on_observation` guards its own body, but its prologue (instrument
    lookup, path construction, `validate_force`, logger setup) runs before that
    guard, so the batch loop needs its own.
    """

    @staticmethod
    def _observation(main_id):
        observation = MagicMock()
        observation.observation = {
            "MAIN_ID": [main_id],
            "FILTER": ["DB_K12"],
            "NIGHT_START": ["2015-09-24"],
        }
        return observation

    def test_failing_observation_does_not_abort_the_batch(self, monkeypatch):
        from spherical.pipeline import run_trap

        first, second = self._observation("51 Eri"), self._observation("bet Pic")
        seen = []

        def fake_run(observation, **_kwargs):
            seen.append(observation)
            if observation is first:
                raise RuntimeError("prologue blew up before the logger existed")

        monkeypatch.setattr(run_trap, "run_trap_on_observation", fake_run)

        run_trap.run_trap_on_observations(
            observations=[first, second],
            trap_config=MagicMock(),
            reduction_config=MagicMock(),
            species_database_directory="/tmp/species",
        )

        assert seen == [first, second]


class TestDescribeObservation:
    def test_builds_target_band_night_label(self):
        from spherical.pipeline.run_trap import _describe_observation

        observation = MagicMock()
        observation.observation = {
            "MAIN_ID": ["51 Eri"],
            "FILTER": ["DB_K12"],
            "NIGHT_START": ["2015-09-24"],
        }
        assert _describe_observation(observation) == "51_Eri/DB_K12/2015-09-24"

    def test_missing_columns_do_not_raise(self):
        from spherical.pipeline.run_trap import _describe_observation

        observation = MagicMock()
        observation.observation = {}
        assert _describe_observation(observation) == "?/?/?"

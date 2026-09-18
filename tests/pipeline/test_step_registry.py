"""Tests for the shared step-layout helpers in ``spherical.pipeline.step_registry``."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

from spherical.pipeline.step_registry import target_folder_string, trap_result_folder


class TestTargetFolderString:
    def test_spaces_become_underscores(self):
        assert target_folder_string("bet Pic", "DB_K12", "2014-12-07") == "bet_Pic/DB_K12/2014-12-07"

    def test_repeated_whitespace_collapses(self):
        assert target_folder_string("HD   3795", "DB_H23", "2024-06-17") == "HD_3795/DB_H23/2024-06-17"

    def test_name_without_spaces_is_unchanged(self):
        assert target_folder_string("51Eri", "OBS_YJ", "2015-09-25") == "51Eri/OBS_YJ/2015-09-25"


class TestTrapResultFolder:
    def test_ifs_layout(self):
        folder = trap_result_folder("/data/red", "bet Pic", "OBS_YJ", "2015-01-01", instrument="IFS")
        assert str(folder) == "/data/red/IFS/trap/bet_Pic/OBS_YJ/2015-01-01"

    def test_irdis_is_the_default_instrument(self):
        folder = trap_result_folder("/data/red", "HD 3795", "DB_H23", "2024-06-17")
        assert str(folder) == "/data/red/IRDIS/trap/HD_3795/DB_H23/2024-06-17"

    def test_accepts_a_path_reduction_directory(self):
        from pathlib import Path

        folder = trap_result_folder(Path("/data/red"), "HD 3795", "DB_H23", "2024-06-17")
        assert str(folder) == "/data/red/IRDIS/trap/HD_3795/DB_H23/2024-06-17"


class TestToolboxWrapper:
    def test_make_target_folder_string_delegates_unchanged(self):
        from spherical.pipeline.toolbox import make_target_folder_string

        observation = MagicMock()
        observation.observation = {
            "MAIN_ID": ["bet Pic"],
            "FILTER": ["DB_K12"],
            "NIGHT_START": ["2014-12-07"],
        }
        assert make_target_folder_string(observation) == "bet_Pic/DB_K12/2014-12-07"


def test_module_imports_without_the_pipeline_extra():
    """The layout helpers must be usable from a base install (spec section 1.2)."""
    probe = (
        "import sys; import spherical.pipeline.step_registry as sr; "
        "heavy = [m for m in ('trap', 'charis', 'scipy', 'matplotlib', 'photutils') if m in sys.modules]; "
        "assert not heavy, heavy; "
        "print(sr.trap_result_folder('/r', 'HD 3795', 'DB_H23', '2024-06-17'))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/r/IRDIS/trap/HD_3795/DB_H23/2024-06-17"


class TestLeafSteps:
    def _order_and_registry(self):
        from spherical.pipeline.step_registry import StepSpec

        registry = {
            "early": StepSpec("early", lambda d: []),
            "leafy": StepSpec("leafy", lambda d: [], leaf=True),
            "late": StepSpec("late", lambda d: []),
        }
        return list(registry), registry

    def test_forcing_a_leaf_forces_only_itself(self):
        from spherical.pipeline.step_registry import _forced

        order, registry = self._order_and_registry()
        force = {"leafy"}
        assert _forced("leafy", force, step_order=order, registry=registry) is True
        assert _forced("late", force, step_order=order, registry=registry) is False
        assert _forced("early", force, step_order=order, registry=registry) is False

    def test_forcing_an_earlier_step_still_forces_the_leaf(self):
        from spherical.pipeline.step_registry import _forced

        order, registry = self._order_and_registry()
        force = {"early"}
        assert _forced("leafy", force, step_order=order, registry=registry) is True
        assert _forced("late", force, step_order=order, registry=registry) is True

    def test_leaf_and_non_leaf_forced_together(self):
        """The non-leaf still sets the cascade start; the leaf does not lower it."""
        from spherical.pipeline.step_registry import _forced

        order, registry = self._order_and_registry()
        force = {"leafy", "late"}
        assert _forced("early", force, step_order=order, registry=registry) is False
        assert _forced("leafy", force, step_order=order, registry=registry) is True
        assert _forced("late", force, step_order=order, registry=registry) is True

    def test_only_leaves_forced_does_not_cascade(self):
        from spherical.pipeline.step_registry import StepSpec, _forced

        registry = {
            "a": StepSpec("a", lambda d: []),
            "b": StepSpec("b", lambda d: [], leaf=True),
            "c": StepSpec("c", lambda d: [], leaf=True),
            "d": StepSpec("d", lambda d: []),
        }
        order = list(registry)
        force = {"b", "c"}
        assert _forced("a", force, step_order=order, registry=registry) is False
        assert _forced("b", force, step_order=order, registry=registry) is True
        assert _forced("c", force, step_order=order, registry=registry) is True
        assert _forced("d", force, step_order=order, registry=registry) is False

    def test_force_true_still_forces_leaves(self):
        from spherical.pipeline.step_registry import _forced

        order, registry = self._order_and_registry()
        assert _forced("leafy", True, step_order=order, registry=registry) is True

    def test_step_absent_from_the_registry_is_treated_as_non_leaf(self):
        """Callers pass an instrument step_order with the default registry."""
        from spherical.pipeline.step_registry import IRDIS_STEP_ORDER, _forced

        assert _forced("preprocess_irdis", {"irdis_calibration"}, step_order=IRDIS_STEP_ORDER) is True

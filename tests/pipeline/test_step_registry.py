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

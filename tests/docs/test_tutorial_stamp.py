"""The line at the top of each tutorial says when, with what and on which machine it ran."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
sys.path.insert(0, str(ROOT / "docs" / "tools"))
import stamp_notebook  # noqa: E402
import tutorial_stamp  # noqa: E402

RUN = {"spherical": "v3.2.0", "charis": "2.1.0", "trap": "2.1.0", "date": "2026-10-20",
       "cpu": "Apple M2 Pro", "cores": 12, "ram_gb": 32.0, "os": "Darwin 24.6.0", "ncpu": 8}


def test_keys_match_stamp_script():
    assert tutorial_stamp.REQUIRED_KEYS == stamp_notebook.PROVENANCE_KEYS


def test_same_day_reads_verified():
    assert tutorial_stamp.format_stamp({"run": RUN, "rendered": "2026-10-20"}) == (
        "Verified with spherical v3.2.0 (charis 2.1.0, TRAP 2.1.0) on 2026-10-20, "
        "on Apple M2 Pro with 12 cores and 32 GB RAM, set_ncpu(8).")


def test_later_render_names_both_dates():
    text = tutorial_stamp.format_stamp({"run": RUN, "rendered": "2027-01-05"})
    assert text.startswith("Reduced with spherical v3.2.0 (charis 2.1.0, TRAP 2.1.0) on 2026-10-20")
    assert text.endswith("Figures rendered on 2027-01-05.")


def test_requires_metadata():
    with pytest.raises(ValueError, match="docs-tutorials"):
        tutorial_stamp.format_stamp(None)


@pytest.mark.parametrize("key", ["spherical", "date"])
def test_requires_keys(key):
    run = {k: v for k, v in RUN.items() if k != key}
    with pytest.raises(ValueError, match=key):
        tutorial_stamp.format_stamp({"run": run, "rendered": "2026-10-20"})


def test_unknown_ram_and_ncpu_are_left_out():
    text = tutorial_stamp.format_stamp({"run": RUN | {"ram_gb": None, "ncpu": None}, "rendered": "2026-10-20"})
    assert "GB RAM" not in text and "set_ncpu" not in text

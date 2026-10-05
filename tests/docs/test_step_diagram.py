"""The Anatomy step diagram covers every registered step, in order."""

import sys
from pathlib import Path

import pytest

from spherical.pipeline import step_registry

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "_ext"))
import step_diagram  # noqa: E402

IFS = step_registry.STEP_ORDER
IRDIS = step_registry.IRDIS_STEP_ORDER


def test_real_registries_are_fully_placed():
    rows = step_diagram.build_phase_rows(IFS, IRDIS)
    placed_ifs = [step for row in rows for step in row.ifs]
    placed_irdis = [step for row in rows for step in row.irdis]
    assert placed_ifs == IFS
    assert placed_irdis == IRDIS


def test_unplaced_step_raises():
    with pytest.raises(ValueError, match="new_step"):
        step_diagram.build_phase_rows([*IFS, "new_step"], IRDIS)


def test_phase_order_must_follow_step_order():
    swapped = list(IFS)
    a, b = swapped.index("find_centers"), swapped.index("extract_cubes")
    swapped[a], swapped[b] = swapped[b], swapped[a]
    with pytest.raises(ValueError, match="order"):
        step_diagram.build_phase_rows(swapped, IRDIS)


def test_identical_lanes_render_once():
    rows = step_diagram.build_phase_rows(IFS, IRDIS)
    by_name = {row.name: row for row in rows}
    assert by_name["Find the star"].shared
    assert not by_name["Calibrate and build cubes"].shared


def test_tools_are_labelled():
    html = step_diagram.render_step_diagram(step_diagram.build_phase_rows(IFS, IRDIS), optional=set())
    assert html.count(">charis<") == 2
    assert html.count(">TRAP<") == 2


def test_optional_steps_are_marked():
    html = step_diagram.render_step_diagram(step_diagram.build_phase_rows(IFS, IRDIS), optional={"align_frames"})
    assert html.count("step-diagram__step--optional") == 1


def test_every_step_name_appears_as_code():
    html = step_diagram.render_step_diagram(step_diagram.build_phase_rows(IFS, IRDIS), optional=set())
    for step in set(IFS) | set(IRDIS):
        assert f"<code>{step}</code>" in html

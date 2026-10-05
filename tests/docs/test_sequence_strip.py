"""The landing-page sequence strip renders every stage as a link, in order."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "_ext"))
import sequence_strip  # noqa: E402

HREFS = {key: f"{key}.html" for key, _ in sequence_strip.STAGES}


def test_stages_render_in_order_as_links():
    html = sequence_strip.render_sequence_strip(HREFS)
    positions = [html.index(f'href="{key}.html"') for key, _ in sequence_strip.STAGES]
    assert positions == sorted(positions)
    for _, label in sequence_strip.STAGES:
        assert f">{label}</a>" in html


def test_frames_follow_the_sphere_sequence_with_coro_widest():
    html = sequence_strip.render_sequence_strip(HREFS)
    labels = [label for label, _ in sequence_strip.FRAMES]
    assert labels == ["FLUX", "CENTER", "CORO", "CENTER", "FLUX"]
    assert max(sequence_strip.FRAMES, key=lambda frame: frame[1])[0] == "CORO"
    assert html.count("sequence-strip__frame ") == 5


def test_current_stage_is_marked_for_assistive_technology():
    html = sequence_strip.render_sequence_strip(HREFS, current="trap")
    assert html.count('aria-current="page"') == 1
    assert 'href="trap.html" aria-current="page"' in html


def test_missing_stage_link_raises():
    hrefs = dict(HREFS)
    del hrefs["products"]
    with pytest.raises(ValueError, match="products"):
        sequence_strip.render_sequence_strip(hrefs)


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="calibration"):
        sequence_strip.render_sequence_strip({**HREFS, "calibration": "x.html"})
    with pytest.raises(ValueError, match="nowhere"):
        sequence_strip.render_sequence_strip(HREFS, current="nowhere")


def test_hrefs_are_escaped():
    html = sequence_strip.render_sequence_strip({**HREFS, "archive": 'a.html"><script>'})
    assert "<script>" not in html

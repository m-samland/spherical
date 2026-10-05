"""The big-picture map has one entry per strip stage, each linked and described."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "_ext"))
sys.path.insert(0, str(Path(__file__).parent))
import pipeline_map  # noqa: E402
import sequence_strip  # noqa: E402
from test_prose_rules import prose_violations  # noqa: E402

HREFS = {key: f"{key}.html" for key, _ in sequence_strip.STAGES}


def test_details_cover_exactly_the_strip_stages():
    assert list(pipeline_map.STAGE_DETAILS) == [key for key, _ in sequence_strip.STAGES]


def test_stages_render_in_order_with_links():
    html = pipeline_map.render_pipeline_map(HREFS)
    positions = [html.index(f'href="{key}.html"') for key, _ in sequence_strip.STAGES]
    assert positions == sorted(positions)


def test_pipeline_map_needs_every_stage():
    hrefs = dict(HREFS)
    del hrefs["trap"]
    with pytest.raises(ValueError, match="trap"):
        pipeline_map.render_pipeline_map(hrefs)


def test_hrefs_are_escaped():
    assert "<script>" not in pipeline_map.render_pipeline_map({**HREFS, "archive": '"><script>'})


def test_captions_follow_the_prose_rules():
    text = "\n".join(f"{does} {gives}" for does, gives in pipeline_map.STAGE_DETAILS.values())
    assert prose_violations(text) == []

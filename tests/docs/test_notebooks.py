"""Committed notebooks stay small, stamped, private and free of output the docs cannot show."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
sys.path.insert(0, str(Path(__file__).parent))
import test_prose_rules as prose_rules  # noqa: E402  (plain import: its tests are not collected twice)
import tutorial_stamp  # noqa: E402

DOCS = ROOT / "docs"
NOTEBOOKS = sorted(p for p in DOCS.rglob("*.ipynb")
                   if "_build" not in p.parts and ".ipynb_checkpoints" not in p.parts)
TUTORIALS = [p for p in NOTEBOOKS if p.parent.name == "tutorials"]
MAX_BYTES = 3 * 1024 * 1024
GLUE_TEXT = "application/papermill.record/text/plain"
ALLOWED_MIME = {"text/plain", "text/html", "text/markdown", "image/png",
                GLUE_TEXT, "application/papermill.record/image/png"}
MAX_STREAM_LINES = 60
MAX_TABLE_ROWS = 20
HOME_PATTERN = re.compile(r"/Users/[^/\s]+|/home/[^/\s]+|/u/[^/\s]+|C:\\\\Users\\\\")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _outputs(nb):
    for cell in nb["cells"]:
        yield from cell.get("outputs", [])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_size_budget(path):
    assert path.stat().st_size <= MAX_BYTES


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.name)
def test_tutorials_are_stamped(path):
    nb = _load(path)
    tutorial_stamp.format_stamp(nb["metadata"].get(tutorial_stamp.METADATA_KEY))
    assert any("{tutorial-stamp}" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_output_types_and_lengths(path):
    nb = _load(path)
    assert "widgets" not in nb["metadata"]
    for out in _outputs(nb):
        assert out["output_type"] != "error", "committed notebook contains a traceback"
        if out["output_type"] == "stream":
            assert len("".join(out["text"]).splitlines()) <= MAX_STREAM_LINES
        else:
            assert set(out.get("data", {})) <= ALLOWED_MIME
            assert "".join(out.get("data", {}).get("text/html", "")).count("<tr") <= MAX_TABLE_ROWS + 2


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_glued_values_are_plain(path):
    for out in _outputs(_load(path)):
        value = "".join(out.get("data", {}).get(GLUE_TEXT, ""))
        assert not value.startswith("np."), f"glue a float, int or str, not {value}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_outputs_have_no_home_paths(path):
    assert not HOME_PATTERN.search(json.dumps(list(_outputs(_load(path)))))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_markdown_cells_follow_prose_rules(path):
    nb = _load(path)
    text = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert prose_rules.prose_violations(text) == []


def test_guards_reject_a_bad_notebook(tmp_path):
    """The checks above fail on a notebook with each kind of problem."""
    bad = {"cells": [{"cell_type": "code", "metadata": {}, "source": [], "outputs": [
        {"output_type": "stream", "name": "stdout", "text": "x\n" * (MAX_STREAM_LINES + 1)},
        {"output_type": "display_data", "metadata": {}, "data": {GLUE_TEXT: "np.float64(1.0)"}},
        {"output_type": "stream", "name": "stdout", "text": "/Users/someone/data\n"}]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "bad.ipynb"
    path.write_text(json.dumps(bad))
    for check in (test_output_types_and_lengths, test_glued_values_are_plain, test_outputs_have_no_home_paths):
        with pytest.raises(AssertionError):
            check(path)
    with pytest.raises(ValueError):
        test_tutorials_are_stamped(path)

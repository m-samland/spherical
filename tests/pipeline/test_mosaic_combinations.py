"""Tests for combination-restriction plumbing in mosaic file discovery.

These lock in the behaviour that the batched plotting helpers rely on: passing
an explicit ``combinations`` list restricts file discovery to that subset,
which replaces the old ``globals()`` monkeypatch of
``get_mosaic_file_combinations``.
"""
import inspect
from pathlib import Path

from spherical.pipeline.visualize import mosaic


def _make_tree(base: Path, combos, template="flat"):
    """Create empty FITS/CSV files for each (target, mode, date) combination."""
    fits_name = mosaic.TEMPLATE_PATTERNS[template]
    csv_name = mosaic.CANDIDATE_PATTERNS[template]
    for target, mode, date in combos:
        leaf = base / target / mode / date
        for name in (fits_name, csv_name):
            file_path = leaf / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("")


def test_get_combinations_defaults_to_all(tmp_path):
    combos = [("HD1", "modeA", "2020-01-01"), ("HD2", "modeA", "2020-01-02")]
    _make_tree(tmp_path, combos)

    result = mosaic.get_mosaic_file_combinations(tmp_path, "flat", "fits")

    assert set(result.keys()) == set(combos)
    assert all(v is not None for v in result.values())


def test_get_combinations_restricts_to_subset(tmp_path):
    combos = [("HD1", "modeA", "2020-01-01"), ("HD2", "modeA", "2020-01-02")]
    _make_tree(tmp_path, combos)
    subset = [combos[0]]

    result = mosaic.get_mosaic_file_combinations(
        tmp_path, "flat", "fits", combinations=subset
    )

    assert set(result.keys()) == set(subset)


def test_single_plot_functions_accept_combinations():
    for name in ("plot_detection_mosaic", "plot_spectrum_mosaic", "plot_combined_mosaic"):
        sig = inspect.signature(getattr(mosaic, name))
        assert "combinations" in sig.parameters, f"{name} missing combinations param"


def test_no_globals_monkeypatch_remains():
    source = inspect.getsource(mosaic)
    assert "globals()['get_mosaic_file_combinations']" not in source
    assert "_get_batch_file_combinations" not in source


def test_build_batch_filename_basic():
    assert (
        mosaic._build_batch_filename("combined", "flat", None, 0, 4, "png")
        == "combined_mosaic_flat_batch_01_of_04.png"
    )


def test_build_batch_filename_suffix_and_pdf():
    assert (
        mosaic._build_batch_filename("detection", "L-type", "young", 2, 4, "pdf")
        == "detection_mosaic_L-type_young_batch_03_of_04.pdf"
    )


def test_batched_functions_accept_format_and_suffix():
    for name in (
        "plot_detection_mosaic_batched",
        "plot_spectrum_mosaic_batched",
        "plot_combined_mosaic_batched",
    ):
        sig = inspect.signature(getattr(mosaic, name))
        assert "output_format" in sig.parameters, f"{name} missing output_format"
        assert "suffix" in sig.parameters, f"{name} missing suffix"

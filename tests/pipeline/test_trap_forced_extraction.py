"""Forced TRAP extraction at known companion positions."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("trap")

from spherical.pipeline import trap_forced_extraction as tfe  # noqa: E402

TRAP_COLUMNS = ["candidate_id", "wavelength_index", "wavelength", "contrast", "uncertainty", "snr",
                "original_unc", "norm_factor"]


class FakeAnalysis:
    instances = []

    def __init__(self, reduction_parameters=None, instrument=None):
        self.reduction_parameters = FakeParameters()
        self.read_output_args = None
        self.extract_kwargs = None
        FakeAnalysis.instances.append(self)

    def read_output(self, component_fraction, **kwargs):
        self.read_output_args = (component_fraction, kwargs)
        self.calls = ["read_output"]

    def contrast_table_and_normalization(self, save=False, inplace=True):
        self.calls.append(("normalization", save, inplace))
        table = pd.DataFrame({"sep (pix)": np.arange(1.0, 80.0), "snr_normalization": np.ones(79)})
        table.loc[table["sep (pix)"] < 5, "snr_normalization"] = np.nan
        self.detection_products = {"contrast_tables": [table, table]}

    def extract_candidate_spectra(self, temporal_components_fraction, data_full, flux_psf_full, pa, **kwargs):
        self.extract_kwargs = kwargs
        self.calls.append("extract")
        rows = []
        for index, _ in enumerate(kwargs["yx_candidate_positions"]):
            for channel in range(2):
                rows.append([index, channel, 2.1 + 0.1 * channel, 1e-5, 1e-6, 10.0, 1e-6, 1.0])
        return pd.DataFrame(rows, columns=TRAP_COLUMNS)


class FakeParameters:
    def merge(self, **kwargs):
        return self


@pytest.fixture
def run(tmp_path, monkeypatch):
    FakeAnalysis.instances = []
    monkeypatch.setattr(tfe, "DetectionAnalysis", FakeAnalysis)
    positions = np.array([[-36.11, -8.55], [10.0, 20.0]])

    def call():
        return tfe.extract_forced_spectra(
            tmp_path, 0.15, positions, data_full=np.zeros((2, 2, 5, 5)), flux_psf_full=np.zeros((2, 5, 5)),
            pa=np.zeros(2), wavelength_indices=np.array([0, 1]),
        )

    return call, positions


def test_reads_fresh_analysis_and_normalizes_before_extracting(run, tmp_path):
    call, positions = run
    call()
    assert len(FakeAnalysis.instances) == 1
    analysis = FakeAnalysis.instances[0]
    fraction, kwargs = analysis.read_output_args
    assert fraction == 0.15
    assert kwargs["result_folder"] == str(tmp_path)
    assert analysis.calls == ["read_output", ("normalization", False, True), "extract"]
    np.testing.assert_array_equal(analysis.extract_kwargs["yx_candidate_positions"], positions)
    assert analysis.extract_kwargs["return_spectra"] is True


def test_writes_csv_with_position_columns(run, tmp_path):
    call, positions = run
    call()
    table = pd.read_csv(tfe.forced_spectra_path(tmp_path, 0.15))
    for column in TRAP_COLUMNS + ["y_relative", "x_relative"]:
        assert column in table.columns
    second = table[table["candidate_id"] == 1]
    assert (second["y_relative"].iloc[0], second["x_relative"].iloc[0]) == tuple(positions[1])


def test_refuses_position_outside_reduced_separations(tmp_path, monkeypatch):
    FakeAnalysis.instances = []
    monkeypatch.setattr(tfe, "DetectionAnalysis", FakeAnalysis)
    with pytest.raises(ValueError, match="outside the separations"):
        tfe.extract_forced_spectra(
            tmp_path, 0.15, np.array([[95.0, 0.0]]), data_full=None, flux_psf_full=None, pa=None,
            wavelength_indices=np.array([0, 1]),
        )
    assert FakeAnalysis.instances[0].calls == ["read_output", ("normalization", False, True)]


def test_forced_run_name_changes_with_positions():
    a = tfe.forced_extraction_run(0.15, np.array([[-59.03, -13.84]]))
    b = tfe.forced_extraction_run(0.15, np.array([[-59.03, -13.0]]))
    assert a != b
    assert a == tfe.forced_extraction_run(0.15, np.array([[-59.03, -13.84]]))
    assert a.startswith("frac0.15_temporal")


def test_forced_step_skips_without_positions(tmp_path, caplog):
    import logging

    from spherical.pipeline import run_trap
    from spherical.pipeline.step_registry import StepDirs, marker_for

    logger = logging.getLogger("test_forced_step")
    with caplog.at_level(logging.WARNING, logger="test_forced_step"):
        run_trap._run_forced_extraction_step(
            yx_known_companion_position=None,
            result_folder=tmp_path,
            trap_dirs=StepDirs(trap_result_folder=tmp_path),
            force=False,
            logger=logger,
            instrument="IFS",
            temporal_components_fraction=0.15,
            inputs={},
        )
    assert not marker_for("run_trap_forced_extraction", tmp_path).exists()
    assert "no yx_known_companion_position configured" in caplog.text


def test_forced_step_runs_and_writes_marker(tmp_path, monkeypatch):
    import logging

    from spherical.pipeline import run_trap
    from spherical.pipeline.step_registry import StepDirs, marker_for

    calls = []
    monkeypatch.setattr(run_trap, "extract_forced_spectra", lambda *args, **kwargs: calls.append((args, kwargs)))
    positions = [-59.03, -13.84]
    kwargs = dict(
        yx_known_companion_position=positions, result_folder=tmp_path,
        trap_dirs=StepDirs(trap_result_folder=tmp_path), force=False, logger=logging.getLogger("t"),
        instrument="IFS", temporal_components_fraction=0.15, inputs={"data_full": None},
    )
    run_trap._run_forced_extraction_step(**kwargs)
    marker = marker_for("run_trap_forced_extraction", tmp_path)
    assert marker.read_text() == tfe.forced_extraction_run(0.15, np.array([positions]))
    # Complete for the same positions: no second extraction.
    run_trap._run_forced_extraction_step(**kwargs)
    assert len(calls) == 1
    # New positions discard the marker and extract again.
    kwargs["yx_known_companion_position"] = [-59.0, -13.0]
    run_trap._run_forced_extraction_step(**kwargs)
    assert len(calls) == 2

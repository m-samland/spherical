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

    def extract_candidate_spectra(self, temporal_components_fraction, data_full, flux_psf_full, pa, **kwargs):
        self.extract_kwargs = kwargs
        rows = []
        for index, _ in enumerate(kwargs["yx_candidate_positions"]):
            for channel in range(2):
                rows.append([index, channel, 2.1 + 0.1 * channel, 1e-5, 1e-6, 10.0, 1e-6, 1.0])
        return pd.DataFrame(rows, columns=TRAP_COLUMNS)


class FakeParameters:
    def merge(self, **kwargs):
        return self


@pytest.fixture
def run(tmp_path):
    FakeAnalysis.instances = []
    positions = np.array([[-36.11, -8.55], [10.0, 20.0]])

    def call():
        return tfe.extract_forced_spectra(
            tmp_path, 0.15, positions, data_full=np.zeros((2, 2, 5, 5)), flux_psf_full=np.zeros((2, 5, 5)),
            pa=np.zeros(2), wavelength_indices=np.array([0, 1]), analysis_factory=FakeAnalysis,
        )

    return call, positions


def test_uses_fresh_analysis_and_reads_output(run, tmp_path):
    call, _ = run
    call()
    assert len(FakeAnalysis.instances) == 1
    fraction, kwargs = FakeAnalysis.instances[0].read_output_args
    assert fraction == 0.15
    assert kwargs["result_folder"] == str(tmp_path)


def test_passes_positions_and_returns_spectra(run):
    call, positions = run
    call()
    kwargs = FakeAnalysis.instances[0].extract_kwargs
    np.testing.assert_array_equal(kwargs["yx_candidate_positions"], positions)
    assert kwargs["return_spectra"] is True


def test_writes_csv_with_position_columns(run, tmp_path):
    call, _ = run
    call()
    table = pd.read_csv(tfe.forced_spectra_path(tmp_path, 0.15))
    for column in TRAP_COLUMNS + ["position_index", "y_relative", "x_relative", "separation_px"]:
        assert column in table.columns
    first = table[table["position_index"] == 0]
    assert first["separation_px"].iloc[0] == pytest.approx(37.11, abs=0.01)


def test_forced_run_name_changes_with_positions():
    a = tfe.forced_extraction_run(0.15, np.array([[-59.03, -13.84]]))
    b = tfe.forced_extraction_run(0.15, np.array([[-59.03, -13.0]]))
    assert a != b
    assert a == tfe.forced_extraction_run(0.15, np.array([[-59.03, -13.84]]))
    assert a.startswith("frac0.15_temporal")

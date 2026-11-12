"""run_trap passes the star amplitude of star_amplitude.fits and the PSF of its usable blocks to TRAP."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

pytest.importorskip("trap")

N_WAVE, N_FRAMES, N_BLOCKS, SIZE = 2, 5, 2, 7


def _converted(directory, star_amplitude=None, blocks=(1, 0), spot_trace=None, science="coro"):
    directory.mkdir(parents=True)
    fits.writeto(directory / "wavelengths.fits", np.array([2110.0, 2251.0]))
    fits.writeto(directory / f"{science}_cube.fits", np.ones((N_WAVE, N_FRAMES, SIZE, SIZE), dtype=np.float32))
    pd.DataFrame({"DEROT ANGLE": np.linspace(0, 20, N_FRAMES)}).to_csv(directory / f"frames_info_{science}.csv", index=False)
    psf = np.stack([np.ones((N_WAVE, SIZE, SIZE)), 3 * np.ones((N_WAVE, SIZE, SIZE))], axis=1)
    fits.writeto(directory / "psf_cube_for_postprocessing.fits", psf)
    fits.writeto(directory / "image_centers_fitted_robust.fits", np.full((N_WAVE, N_FRAMES, 2), 3.0))
    fits.writeto(directory / "badpixel_map.fits", np.zeros((N_WAVE, SIZE, SIZE), dtype=np.uint8))
    if spot_trace is not None:
        fits.writeto(directory / "spot_amplitude_variation.fits", spot_trace)
    if star_amplitude is not None:
        from spherical.pipeline.steps.star_amplitude_calibration import _write

        header = {"MODE": "wfs", "REASON": "variable", "LEVOFF": None, "DROPDEP": None, "DISAGREE": None,
                  "SCALEUNC": None, "NFLOOR": 1, "NFILLED": 0, "MAXGAP_S": 0.0, "SEGMENT": None, "WFSFLOOR": 0.1,
                  "WAFMODE": "wfs"}
        _write(directory, star_amplitude, header, np.asarray(blocks), np.ones(N_WAVE), np.ones(N_WAVE))


def _run(monkeypatch, tmp_path, *, use_star_amplitude=True, pass_amplitude_modulation=False, waffle=False):
    from spherical.pipeline import run_trap
    from spherical.pipeline.pipeline_config import IRDISReductionConfig

    captured = {}
    monkeypatch.setattr(run_trap, "_data_directory_for", lambda *_: str(tmp_path / "converted"))
    monkeypatch.setattr(run_trap, "run_complete_reduction", lambda **kwargs: captured.update(kwargs))

    observation = MagicMock()
    observation.observation = {"INSTRUMENT": ["IRDIS"], "MAIN_ID": ["51 Eri"], "FILTER": ["DB_K12"],
                               "NIGHT_START": ["2015-09-24"], "WAFFLE_MODE": [waffle]}
    trap_config = MagicMock()
    trap_config.get_instrument.return_value.instrument_type = "ifu"
    trap_config.detection.template_names = None
    trap_config.processing.verbose = False
    trap_config.processing.temporal_components_fraction = [0.3]
    trap_config.processing.wavelength_indices = [0, 1]

    reduction_config = IRDISReductionConfig()
    reduction_config.directories.reduction_directory = tmp_path / "reduction"
    reduction_config.use_gaia_stellar_parameters = False
    reduction_config.use_star_amplitude = use_star_amplitude
    reduction_config.pass_amplitude_modulation_to_trap = pass_amplitude_modulation
    reduction_config.steps = reduction_config.steps.merge(run_trap_detection=False, run_trap_forced_extraction=False)

    run_trap.run_trap_on_observation(observation=observation, trap_config=trap_config,
                                     reduction_config=reduction_config, species_database_directory=tmp_path)
    assert captured, "run_complete_reduction was not reached"
    return captured


def test_trap_gets_the_star_amplitude_and_the_psf_of_the_usable_blocks(monkeypatch, tmp_path):
    amplitude = np.array([[1.0, 0.9, np.nan, 1.1, 1.0], [1.0, 0.8, 0.7, 1.2, 1.0]])
    _converted(tmp_path / "converted", star_amplitude=amplitude, blocks=(1, 0))
    kwargs = _run(monkeypatch, tmp_path)
    expected = amplitude.copy()
    expected[:, 2] = 1.0
    np.testing.assert_allclose(kwargs["amplitude_modulation_full"], expected.astype(np.float32))
    assert kwargs["bad_frames"] == [2]
    np.testing.assert_allclose(kwargs["flux_psf_full"], np.ones((N_WAVE, SIZE, SIZE)))


def test_star_amplitude_replaces_the_spot_trace(monkeypatch, tmp_path):
    amplitude = np.full((N_WAVE, N_FRAMES), 0.5)
    spot_trace = np.full((N_WAVE, N_FRAMES), 2.0)
    _converted(tmp_path / "converted", star_amplitude=amplitude, spot_trace=spot_trace,
               science="center")
    kwargs = _run(monkeypatch, tmp_path, pass_amplitude_modulation=True, waffle=True)
    np.testing.assert_allclose(kwargs["amplitude_modulation_full"], amplitude)


def test_without_use_star_amplitude_trap_gets_todays_inputs(monkeypatch, tmp_path):
    _converted(tmp_path / "converted", star_amplitude=np.full((N_WAVE, N_FRAMES), 0.5))
    kwargs = _run(monkeypatch, tmp_path, use_star_amplitude=False)
    assert kwargs["amplitude_modulation_full"] is None
    assert kwargs["bad_frames"] is None
    np.testing.assert_allclose(kwargs["flux_psf_full"], 2 * np.ones((N_WAVE, SIZE, SIZE)))


def test_a_channel_without_amplitude_does_not_make_every_frame_bad(monkeypatch, tmp_path):
    amplitude = np.array([[np.nan] * N_FRAMES, [1.0, 0.8, np.nan, 1.2, 1.0]])
    _converted(tmp_path / "converted", star_amplitude=amplitude, blocks=(1, 0))
    kwargs = _run(monkeypatch, tmp_path)
    assert kwargs["bad_frames"] == [2]
    np.testing.assert_array_equal(kwargs["amplitude_modulation_full"][0], np.ones(N_FRAMES))

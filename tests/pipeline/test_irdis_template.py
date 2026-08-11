"""Tests for IRDIS → species filter mapping and SpectralTemplate photometry.

The mapping test is pure Python and always runs. The SpectralTemplate smoke
test exercises the TRAP photometry branch end-to-end (species SyntheticPhotometry
+ SVO filter lookup) and is skipped when species / trap / an initialized species
database are unavailable — matching the CI environment described in CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u


class TestSpeciesFiltersForMode:
    def test_all_db_modes_map_to_pair_of_names(self):
        from spherical.pipeline.irdis_filters import (
            IRDIS_SPECIES_FILTERS,
            species_filters_for_mode,
        )

        assert set(IRDIS_SPECIES_FILTERS) == {
            "DB_K12", "DB_H23", "DB_H34", "DB_Y23", "DB_J23"
        }
        for mode in IRDIS_SPECIES_FILTERS:
            pair = species_filters_for_mode(mode)
            assert pair is not None
            assert len(pair) == 2
            assert all(isinstance(name, str) and name for name in pair)
            assert all(name.startswith("Paranal/SPHERE.IRDIS_D_") for name in pair)

    @pytest.mark.parametrize("mode", ["BB_H", "BB_K", "BB_Ks", "NB_BrG", "DP_0_BB_H"])
    def test_single_channel_modes_return_none(self, mode):
        from spherical.pipeline.irdis_filters import species_filters_for_mode

        assert species_filters_for_mode(mode) is None

    def test_unknown_mode_returns_none(self):
        from spherical.pipeline.irdis_filters import species_filters_for_mode

        assert species_filters_for_mode("NOT_A_REAL_MODE") is None


class TestSpectralTemplatePhotometryBranch:
    """Verify TRAP's SpectralTemplate 'photometry' branch accepts a 2-filter
    IRDIS setup and populates ``contrast_modelbox`` without raising.

    This is the exact code path the 51 Eri DB_K12 crash hit. Skipped when
    species / trap or the species DB are not available on the runner.
    """

    def test_dbi_two_filter_smoke(self, tmp_path, monkeypatch):
        pytest.importorskip("species")
        pytest.importorskip("trap")

        import os

        from species import SpeciesInit
        from species.data.database import Database
        from species.read.read_filter import ReadFilter
        from trap.parameters import Instrument
        from trap.template import SpectralTemplate

        monkeypatch.chdir(tmp_path)
        try:
            SpeciesInit()
            database = Database()
            database.add_filter("Paranal/SPHERE.IRDIS_D_K12_1", verbose=False)
            database.add_filter("Paranal/SPHERE.IRDIS_D_K12_2", verbose=False)
            ReadFilter("Paranal/SPHERE.IRDIS_D_K12_1").mean_wavelength()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"species database not usable in this environment: {exc}")
        assert os.getcwd() == str(tmp_path)

        instrument = Instrument(
            name="IRDIS",
            pixel_scale=u.pixel_scale(0.01225 * u.arcsec / u.pixel),
            telescope_diameter=7.99 * u.m,
            instrument_type="photometry",
            wavelengths=np.array([2.11, 2.25]) * u.micron,
            filters=["Paranal/SPHERE.IRDIS_D_K12_1", "Paranal/SPHERE.IRDIS_D_K12_2"],
        )

        wavelength = np.linspace(0.9, 2.4, 200)
        star_box = _flat_modelbox(wavelength, flux=1.0)
        planet_box = _flat_modelbox(wavelength, flux=1e-5)

        template = SpectralTemplate(
            name="test",
            instrument=instrument,
            companion_modelbox=planet_box,
            stellar_modelbox=star_box,
        )

        assert hasattr(template, "contrast_modelbox")
        assert len(template.contrast_modelbox.flux) == 2
        assert np.all(np.isfinite(template.contrast_modelbox.flux))


def _flat_modelbox(wavelength: np.ndarray, flux: float):
    from species.core.box import create_box

    return create_box(
        boxtype="model",
        model="synthetic",
        wavelength=wavelength,
        flux=np.full_like(wavelength, flux, dtype=float),
        parameters={},
        quantity="flux",
    )

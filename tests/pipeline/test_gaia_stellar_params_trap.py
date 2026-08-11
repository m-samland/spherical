"""Tests for the TRAP stellar-parameter resolution in run_trap.py."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

# trap is not installed in the CI `test` env (pipeline deps are validated
# locally only); skip the whole module there instead of erroring at collection.
pytest.importorskip("trap")

from trap.parameters import trap_config_for_ifs

from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.run_trap import (
    _apply_stellar_params,
    _teff_from_spectral_type,
)


def _make_observation(gaia_teff=None, gaia_logg=None, gaia_mh=None, sp_type=None, include_gaia=True):
    """Create a minimal mock observation with optional Gaia and SP_TYPE columns."""
    data = {
        "MAIN_ID": ["HD 12345"],
        "FILTER": ["OBS_YJ"],
    }
    if include_gaia:
        data["GAIA_TEFF"] = [gaia_teff if gaia_teff is not None else np.nan]
        data["GAIA_LOGG"] = [gaia_logg if gaia_logg is not None else np.nan]
        data["GAIA_MH"] = [gaia_mh if gaia_mh is not None else np.nan]
    if sp_type is not None:
        data["SP_TYPE"] = [sp_type]

    return SimpleNamespace(observation=Table(data))


@pytest.fixture
def default_config():
    """Return a default IFS TrapConfig."""
    return trap_config_for_ifs()


@pytest.fixture
def logger():
    return logging.getLogger("test_stellar_params")


# --- Tier 1: Gaia DR3 ------------------------------------------------------

class TestGaiaTier:
    """Finite Gaia values take priority for teff/logg; metallicity stays solar."""

    def test_teff_logg_from_gaia_feh_stays_solar(self, default_config, logger):
        # GAIA_MH is deliberately ignored: [Fe/H] is kept at the configured solar
        # default (negligible for low-res IFS templates; bt-nextgen grid is solar-only).
        default_config.detection.stellar_parameters.feh = 0.0
        obs = _make_observation(gaia_teff=5200.0, gaia_logg=3.8, gaia_mh=0.15)
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 5200.0
        assert sp.logg == pytest.approx(3.8)
        assert sp.feh == 0.0

    def test_gaia_teff_anchors_logg_feh_default_when_nan(self, default_config, logger):
        default_config.detection.stellar_parameters.logg = 4.0
        default_config.detection.stellar_parameters.feh = 0.0
        obs = _make_observation(gaia_teff=6500.0, gaia_logg=np.nan, gaia_mh=np.nan)
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 6500.0
        assert sp.logg == 4.0
        assert sp.feh == 0.0

    def test_gaia_wins_over_spectral_type(self, default_config, logger):
        obs = _make_observation(gaia_teff=6000.0, sp_type="M5V")
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 6000.0

    def test_original_config_not_mutated(self, default_config, logger):
        original_teff = default_config.detection.stellar_parameters.teff
        obs = _make_observation(gaia_teff=3400.0)
        _apply_stellar_params(obs, default_config, logger)
        assert default_config.detection.stellar_parameters.teff == original_teff


# --- Tier 2: spectral-type fallback ---------------------------------------

class TestSpectralTypeTier:
    """When Gaia Teff is absent, SP_TYPE supplies Teff; logg/feh stay default."""

    def test_spectral_type_sets_teff_only(self, default_config, logger):
        default_config.detection.stellar_parameters.logg = 4.0
        default_config.detection.stellar_parameters.feh = 0.0
        obs = _make_observation(gaia_teff=np.nan, sp_type="G2V")
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 5770.0  # Mamajek G2V
        assert sp.logg == 4.0
        assert sp.feh == 0.0

    def test_fractional_subtype_floored(self, default_config, logger):
        obs = _make_observation(gaia_teff=np.nan, sp_type="M4.5V")
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 3210.0  # floored to M4

    def test_used_when_gaia_columns_missing(self, default_config, logger):
        obs = _make_observation(include_gaia=False, sp_type="K0V")
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 5270.0  # Mamajek K0V


# --- Tier 3: configured default --------------------------------------------

class TestDefaultTier:
    """No Gaia Teff and no usable spectral type keeps the configured values."""

    def test_junk_spectral_type_preserves_default(self, default_config, logger):
        default_config.detection.stellar_parameters.teff = 7500.0
        obs = _make_observation(gaia_teff=np.nan, sp_type="DA2.9")  # white dwarf
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 7500.0

    def test_no_columns_preserves_default(self, default_config, logger):
        default_config.detection.stellar_parameters.teff = 9000.0
        obs = _make_observation(include_gaia=False)
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.teff == 9000.0

    def test_radius_distance_never_touched(self, default_config, logger):
        default_config.detection.stellar_parameters.radius = 1.8
        default_config.detection.stellar_parameters.distance = 19.44
        obs = _make_observation(gaia_teff=6000.0, sp_type="G2V")
        sp = _apply_stellar_params(obs, default_config, logger).detection.stellar_parameters
        assert sp.radius == 1.8
        assert sp.distance == 19.44


# --- Spectral-type parser (unit level) -------------------------------------

class TestSpectralTypeParser:
    """Strict leading-type parsing; anything ambiguous returns None."""

    @pytest.mark.parametrize(
        "sp_type, expected",
        [
            ("G2V", 5770.0),
            ("K0V", 5270.0),
            ("M4.5V", 3210.0),    # fractional floored to M4
            ("F8/G0V", 6180.0),   # range -> first/primary type F8
            ("G0V+M2V", 5930.0),  # composite -> primary G0
            ("(K0V)", 5270.0),    # parenthesised
            ("O3V", 44900.0),
            ("M9V", 2380.0),
        ],
    )
    def test_valid_types(self, sp_type, expected):
        assert _teff_from_spectral_type(sp_type) == expected

    @pytest.mark.parametrize("sp_type", ["DA2", "sdB", "C-N5", "WN8", "", "   ", None, "L2"])
    def test_rejected_types(self, sp_type):
        assert _teff_from_spectral_type(sp_type) is None


# --- Opt-out now lives on the spherical reduction config -------------------

class TestOptOutConfig:
    """The Gaia/spectral resolution is gated by IFSReductionConfig, not trap."""

    def test_flag_defaults_true(self):
        assert IFSReductionConfig().use_gaia_stellar_parameters is True

    def test_flag_can_be_disabled(self):
        config = IFSReductionConfig()
        config.use_gaia_stellar_parameters = False
        assert config.use_gaia_stellar_parameters is False

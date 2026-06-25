"""Tests for _apply_gaia_stellar_params helper in run_trap.py."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

# trap is not installed in the CI `test` env (pipeline deps are validated
# locally only); skip the whole module there instead of erroring at collection.
pytest.importorskip("trap")

from trap.parameters import trap_config_for_ifs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_observation(gaia_teff=None, gaia_logg=None, gaia_mh=None, include_columns=True):
    """Create a minimal mock observation with optional Gaia columns."""
    data = {
        "MAIN_ID": ["HD 12345"],
        "FILTER": ["OBS_YJ"],
        "NIGHT_START": ["2024-01-01"],
        "WAFFLE_MODE": ["CORO"],
    }
    if include_columns:
        data["GAIA_TEFF"] = [gaia_teff if gaia_teff is not None else np.nan]
        data["GAIA_LOGG"] = [gaia_logg if gaia_logg is not None else np.nan]
        data["GAIA_MH"] = [gaia_mh if gaia_mh is not None else np.nan]

    obs_table = Table(data)
    return SimpleNamespace(observation=obs_table)


@pytest.fixture
def default_config():
    """Return a default IFS TrapConfig."""
    return trap_config_for_ifs()


@pytest.fixture
def logger():
    return logging.getLogger("test_gaia_stellar_params")


# ---------------------------------------------------------------------------
# Import the helper
# ---------------------------------------------------------------------------

from spherical.pipeline.run_trap import _apply_gaia_stellar_params


# ---------------------------------------------------------------------------
# Tests: basic override
# ---------------------------------------------------------------------------

class TestGaiaOverrideApplied:
    """Gaia values in the observation table should override stellar params."""

    def test_teff_override(self, default_config, logger):
        obs = _make_observation(gaia_teff=6500.0)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.teff == 6500.0

    def test_logg_override(self, default_config, logger):
        obs = _make_observation(gaia_logg=4.35)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.logg == pytest.approx(4.35)

    def test_feh_override(self, default_config, logger):
        obs = _make_observation(gaia_mh=-0.12)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.feh == pytest.approx(-0.12)

    def test_all_three_overridden(self, default_config, logger):
        obs = _make_observation(gaia_teff=5200.0, gaia_logg=3.8, gaia_mh=0.15)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        sp = result.detection.stellar_parameters
        assert sp.teff == 5200.0
        assert sp.logg == pytest.approx(3.8)
        assert sp.feh == pytest.approx(0.15)

    def test_original_config_not_mutated(self, default_config, logger):
        original_teff = default_config.detection.stellar_parameters.teff
        obs = _make_observation(gaia_teff=3400.0)
        _apply_gaia_stellar_params(obs, default_config, logger)
        assert default_config.detection.stellar_parameters.teff == original_teff


# ---------------------------------------------------------------------------
# Tests: fallback when Gaia values are NaN / missing
# ---------------------------------------------------------------------------

class TestFallback:
    """When Gaia values are NaN or columns are absent, defaults are preserved."""

    def test_nan_teff_preserves_default(self, default_config, logger):
        default_config.detection.stellar_parameters.teff = 7500.0
        obs = _make_observation(gaia_teff=np.nan)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.teff == 7500.0

    def test_missing_columns_preserves_default(self, default_config, logger):
        default_config.detection.stellar_parameters.teff = 9000.0
        obs = _make_observation(include_columns=False)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.teff == 9000.0

    def test_partial_nan_only_overrides_valid(self, default_config, logger):
        default_config.detection.stellar_parameters.teff = 8000.0
        default_config.detection.stellar_parameters.logg = 4.0
        obs = _make_observation(gaia_teff=5500.0, gaia_logg=np.nan)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.teff == 5500.0
        assert result.detection.stellar_parameters.logg == 4.0  # unchanged


# ---------------------------------------------------------------------------
# Tests: opt-out flag
# ---------------------------------------------------------------------------

class TestOptOut:
    """Setting use_gaia_stellar_parameters=False should skip override."""

    def test_flag_false_skips_override(self, default_config, logger):
        default_config.detection.use_gaia_stellar_parameters = False
        default_config.detection.stellar_parameters.teff = 8000.0
        obs = _make_observation(gaia_teff=5000.0)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.teff == 8000.0

    def test_flag_false_still_deepcopies(self, default_config, logger):
        default_config.detection.use_gaia_stellar_parameters = False
        obs = _make_observation(gaia_teff=5000.0)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result is not default_config


# ---------------------------------------------------------------------------
# Tests: radius/distance untouched
# ---------------------------------------------------------------------------

class TestUntouchedFields:
    """Fields not in Gaia (radius, distance) should never be modified."""

    def test_radius_unchanged(self, default_config, logger):
        default_config.detection.stellar_parameters.radius = 1.8
        obs = _make_observation(gaia_teff=6000.0, gaia_logg=4.1, gaia_mh=0.0)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.radius == 1.8

    def test_distance_unchanged(self, default_config, logger):
        default_config.detection.stellar_parameters.distance = 19.44
        obs = _make_observation(gaia_teff=6000.0)
        result = _apply_gaia_stellar_params(obs, default_config, logger)
        assert result.detection.stellar_parameters.distance == 19.44

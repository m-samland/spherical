"""Tests for Gaia DR3 astrophysical parameter enrichment.

These tests verify the module against the live ESA Gaia TAP archive using
a small inline target table with known Gaia DR3 identifiers (same fixture
as test_mocadb_matching.py).
"""

import numpy as np
import pytest
from astropy.table import Table

from spherical.database.gaia_astrophysical_params import (
    _clean_gaia_id,
    query_gaia_astrophysical_params,
)

# ---------------------------------------------------------------------------
# Inline test fixture: 4 targets from the SPHERE archive
# ---------------------------------------------------------------------------
# 3 Cen A (B5III, ~17000 K), 4 Sgr (A0, ~10000 K), 5 Vul (A0V, ~10000 K),
# 51 Eri (F0IV, ~7000 K)

@pytest.fixture()
def sample_target_table():
    """Minimal target table with 4 stars, all with valid Gaia DR3 IDs."""
    return Table(
        {
            "MAIN_ID": ["*   3 Cen A", "*   4 Sgr", "*   5 Vul", "51 Eri"],
            "ID_GAIA_DR3": [
                "Gaia DR3 6170485544575679104",
                "Gaia DR3 4069112871640783360",
                "Gaia DR3 4515892996332248960",
                "Gaia DR3 3205095125321700480",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests for _clean_gaia_id
# ---------------------------------------------------------------------------

class TestCleanGaiaId:
    def test_string_with_prefix(self):
        assert _clean_gaia_id("Gaia DR3 6126376695498244736") == 6126376695498244736

    def test_plain_integer_string(self):
        assert _clean_gaia_id("6126376695498244736") == 6126376695498244736

    def test_integer_input(self):
        assert _clean_gaia_id(6126376695498244736) == 6126376695498244736

    def test_numpy_int(self):
        assert _clean_gaia_id(np.int64(6126376695498244736)) == 6126376695498244736

    def test_none(self):
        assert _clean_gaia_id(None) is None

    def test_nan(self):
        assert _clean_gaia_id(float("nan")) is None

    def test_empty_string(self):
        assert _clean_gaia_id("") is None

    def test_dash(self):
        assert _clean_gaia_id("--") is None

    def test_short_number_rejected(self):
        assert _clean_gaia_id("123") is None


# ---------------------------------------------------------------------------
# Integration tests against the live Gaia TAP archive
# ---------------------------------------------------------------------------

@pytest.mark.remote_data
class TestQueryGaiaAstrophysicalParams:

    def test_basic_enrichment(self, sample_target_table):
        """All 12 GAIA_ columns should be present after enrichment."""
        enriched = query_gaia_astrophysical_params(sample_target_table)

        assert len(enriched) == len(sample_target_table)

        expected_cols = [
            "GAIA_TEFF", "GAIA_TEFF_LOWER", "GAIA_TEFF_UPPER",
            "GAIA_LOGG", "GAIA_LOGG_LOWER", "GAIA_LOGG_UPPER",
            "GAIA_MH", "GAIA_MH_LOWER", "GAIA_MH_UPPER",
            "GAIA_AG", "GAIA_AG_LOWER", "GAIA_AG_UPPER",
        ]
        for col in expected_cols:
            assert col in enriched.colnames, f"Missing column: {col}"

    def test_known_star_has_teff(self, sample_target_table):
        """51 Eri (F0IV) should have a Teff around 6000–8000 K from GSP-Phot."""
        enriched = query_gaia_astrophysical_params(sample_target_table)

        # Row 3 = 51 Eri
        teff = enriched[3]["GAIA_TEFF"]
        if np.isnan(teff):
            pytest.skip("Gaia TAP returned no data for 51 Eri (transient issue)")
        assert 4000 < teff < 9000, f"Unexpected Teff for 51 Eri: {teff}"

    def test_teff_has_uncertainties(self, sample_target_table):
        """Stars with Teff should also have lower/upper confidence bounds."""
        enriched = query_gaia_astrophysical_params(sample_target_table)

        # Check any star that has a Teff
        for row in enriched:
            if not np.isnan(row["GAIA_TEFF"]):
                assert not np.isnan(row["GAIA_TEFF_LOWER"]), "Missing TEFF_LOWER"
                assert not np.isnan(row["GAIA_TEFF_UPPER"]), "Missing TEFF_UPPER"
                assert row["GAIA_TEFF_LOWER"] <= row["GAIA_TEFF"] <= row["GAIA_TEFF_UPPER"]
                break
        else:
            pytest.skip("No star had Teff data from Gaia (transient issue)")

    def test_logg_present(self, sample_target_table):
        """At least one star should have logg from GSP-Phot."""
        enriched = query_gaia_astrophysical_params(sample_target_table)

        logg_values = enriched["GAIA_LOGG"]
        if all(np.isnan(logg_values)):
            pytest.skip("No logg data returned (transient issue)")
        valid = logg_values[~np.isnan(logg_values)]
        # logg should be in reasonable range (0–6 for most stars)
        assert all(0 <= v <= 6 for v in valid)

    def test_no_gaia_column_raises(self):
        """Missing ID column should raise ValueError."""
        table = Table({"MAIN_ID": ["test"]})
        with pytest.raises(ValueError, match="ID_GAIA_DR3"):
            query_gaia_astrophysical_params(table)

    def test_all_invalid_ids(self):
        """All invalid IDs should return table with NaN GAIA columns."""
        table = Table({
            "MAIN_ID": ["bad1", "bad2"],
            "ID_GAIA_DR3": ["--", ""],
        })
        enriched = query_gaia_astrophysical_params(table)
        assert len(enriched) == 2
        assert all(np.isnan(enriched["GAIA_TEFF"]))

    def test_idempotent_re_enrichment(self, sample_target_table):
        """Running enrichment twice should not fail or duplicate columns."""
        enriched1 = query_gaia_astrophysical_params(sample_target_table)
        enriched2 = query_gaia_astrophysical_params(enriched1)

        gaia_cols = [c for c in enriched2.colnames if c.startswith("GAIA_")]
        assert len(gaia_cols) == 12
        assert len(enriched2) == len(sample_target_table)

"""Tests for MOCAdb cross-matching module (spherical.database.mocadb_matching).

These tests verify the module against the live MOCAdb MySQL endpoint using
a small inline target table with known Gaia DR3 identifiers.
"""

import numpy as np
import pytest
from astropy.table import Table

from spherical.database.mocadb_matching import (
    _clean_gaia_id,
    query_mocadb_for_targets,
)

# ---------------------------------------------------------------------------
# Inline test fixture: 4 targets from the SPHERE archive
# ---------------------------------------------------------------------------
# 3 Cen A is a known Sco-Cen member, 51 Eri is a Beta Pic member;
# 4 Sgr and 5 Vul are field stars.

@pytest.fixture()
def sample_target_table():
    """Minimal target table with 4 stars, all with valid Gaia DR3 IDs.

    Only ``MAIN_ID`` and ``ID_GAIA_DR3`` are required by
    `query_mocadb_for_targets`; the matching is purely based on
    Gaia DR3 source IDs (no coordinate cross-match).
    """
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
        # source_ids are long; very short numbers should be rejected
        assert _clean_gaia_id("123") is None


# ---------------------------------------------------------------------------
# Integration tests against the live MOCAdb endpoint
# ---------------------------------------------------------------------------

@pytest.mark.remote_data
class TestQueryMOCAdb:
    """Tests that connect to the public MOCAdb MySQL server.

    Marked with ``@pytest.mark.remote_data`` so they can be skipped in
    offline / CI environments via ``pytest -m "not remote_data"``.
    """

    def test_basic_enrichment(self, sample_target_table):
        """All MOCA_ columns are appended and the table length is unchanged."""
        enriched = query_mocadb_for_targets(sample_target_table, include_tier2=True)

        assert len(enriched) == len(sample_target_table)

        expected_tier1 = [
            "MOCA_OID", "MOCA_AID", "MOCA_MEMBERSHIP_TYPE",
            "MOCA_DESIGNATION", "MOCA_SPECTRAL_TYPE",
            "MOCA_BANYAN_PROB", "MOCA_BANYAN_UVW_SEP", "MOCA_YA_PROB",
            "MOCA_ASSOCIATION_NAME", "MOCA_ASSOCIATION_TYPE",
            "MOCA_AGE_MYR", "MOCA_AGE_MYR_UNC",
            "MOCA_AGE_MYR_UNC_POS", "MOCA_AGE_MYR_UNC_NEG",
        ]
        expected_tier2 = [
            "MOCA_SPTN", "MOCA_PARALLAX_MAS",
            "MOCA_X_PC", "MOCA_Y_PC", "MOCA_Z_PC",
            "MOCA_U_KMS", "MOCA_V_KMS", "MOCA_W_KMS",
            "MOCA_PROT_DAYS", "MOCA_GAIA_ACT", "MOCA_EWLI", "MOCA_EWHA",
            "MOCA_DR3_RUWE",
        ]
        for col in expected_tier1 + expected_tier2:
            assert col in enriched.colnames, f"Missing column: {col}"

    def test_known_sco_cen_member(self, sample_target_table):
        """3 Cen A should be identified as a Sco-Cen member with a young age."""
        enriched = query_mocadb_for_targets(sample_target_table, include_tier2=False)

        # Row 0 = 3 Cen A
        row = enriched[0]

        # Skip content assertions if MOCAdb server didn't return data
        # (transient network / server issues).
        if np.isnan(float(row["MOCA_OID"])):
            pytest.skip("MOCAdb server returned no data (transient issue)")

        assert "Sco-Cen" in str(row["MOCA_ASSOCIATION_NAME"]) or \
               "Cen" in str(row["MOCA_ASSOCIATION_NAME"]) or \
               "Ratzenb" in str(row["MOCA_ASSOCIATION_NAME"]), \
               f"Expected Sco-Cen association, got: {row['MOCA_ASSOCIATION_NAME']}"

        age = float(row["MOCA_AGE_MYR"])
        assert 1 < age < 100, f"Expected young age for 3 Cen A, got {age} Myr"

    def test_known_beta_pic_member(self, sample_target_table):
        """51 Eri should be identified as a Beta Pic member with a young age."""
        enriched = query_mocadb_for_targets(sample_target_table, include_tier2=False)

        # Row 3 = 51 Eri
        row = enriched[3]

        if np.isnan(float(row["MOCA_OID"])):
            pytest.skip("MOCAdb server returned no data (transient issue)")

        assoc = str(row["MOCA_ASSOCIATION_NAME"])
        assert "Pic" in assoc or "BPMG" in assoc or "Beta" in assoc, \
            f"Expected Beta Pic association for 51 Eri, got: {assoc}"

        age = float(row["MOCA_AGE_MYR"])
        assert 1 < age < 100, f"Expected young age for 51 Eri, got {age} Myr"

    def test_field_star_empty(self, sample_target_table):
        """Field stars should have NaN / empty MOCA columns."""
        enriched = query_mocadb_for_targets(sample_target_table, include_tier2=False)

        # Rows 1-2 are field stars (4 Sgr, 5 Vul)
        for idx in [1, 2]:
            row = enriched[idx]
            assert np.isnan(float(row["MOCA_AGE_MYR"]))
            assert str(row["MOCA_ASSOCIATION_NAME"]) == ""

    def test_tier2_disabled(self, sample_target_table):
        """When include_tier2=False, tier-2 columns should not be present."""
        enriched = query_mocadb_for_targets(
            sample_target_table, include_tier2=False
        )
        assert "MOCA_AGE_MYR" in enriched.colnames  # tier-1 present
        assert "MOCA_PROT_DAYS" not in enriched.colnames  # tier-2 absent

    def test_no_gaia_column_raises(self, sample_target_table):
        """Passing a table without the Gaia column should raise ValueError."""
        bad_table = sample_target_table.copy()
        bad_table.remove_column("ID_GAIA_DR3")
        with pytest.raises(ValueError, match="ID_GAIA_DR3"):
            query_mocadb_for_targets(bad_table)

    def test_all_invalid_ids(self):
        """Table with no valid Gaia IDs should return empty MOCA columns."""
        bad_table = Table(
            {
                "MAIN_ID": ["star_a", "star_b"],
                "ID_GAIA_DR3": ["--", ""],
            }
        )
        enriched = query_mocadb_for_targets(bad_table, include_tier2=True)
        assert len(enriched) == 2
        assert "MOCA_AGE_MYR" in enriched.colnames
        assert all(np.isnan(float(v)) for v in enriched["MOCA_AGE_MYR"])

"""Tests for enrichment health metrics and threshold evaluation.

No network: all tables are synthetic. Covers metric computation from enriched
columns and the floor / regression / valid-ID-collapse / tier-2 threshold policy.
"""

import numpy as np
from astropy.table import Table

from spherical.database import enrichment_health as eh


def _gaia_table(teff):
    """Target table with ID_GAIA_DR3 + GAIA_TEFF columns.

    ``teff`` entries that are None become an unparseable Gaia ID (no valid ID);
    otherwise the row has a valid ID and the given TEFF (np.nan = matched-miss).
    """
    ids = [f"Gaia DR3 {100000 + i}" if t != "noid" else "" for i, t in enumerate(teff)]
    vals = [np.nan if (t == "noid" or t is None) else float(t) for t in teff]
    return Table({"ID_GAIA_DR3": ids, "GAIA_TEFF": vals})


# --------------------------------------------------------------------------
# compute_source_metrics
# --------------------------------------------------------------------------

class TestComputeGaiaMetrics:
    def test_counts_and_fraction(self):
        # 4 rows: 3 valid IDs, of which 2 have a TEFF; 1 has no Gaia ID.
        tbl = _gaia_table([5000.0, 6000.0, np.nan, "noid"])
        m = eh.compute_source_metrics(tbl, "gaia", status="ok")
        assert m["n_total"] == 4
        assert m["n_valid_ids"] == 3
        assert m["n_matched"] == 2
        assert m["frac"] == 2 / 3

    def test_zero_valid_ids_gives_zero_fraction(self):
        tbl = _gaia_table(["noid", "noid"])
        m = eh.compute_source_metrics(tbl, "gaia", status="ok")
        assert m["n_valid_ids"] == 0
        assert m["frac"] == 0.0


def _moca_table(oid, parallax):
    return Table({
        "ID_GAIA_DR3": [f"Gaia DR3 {100000 + i}" for i in range(len(oid))],
        "MOCA_OID": [np.nan if o is None else float(o) for o in oid],
        "MOCA_PARALLAX_MAS": [np.nan if p is None else float(p) for p in parallax],
    })


class TestComputeMocaTier2:
    def test_tier2_ok_when_any_tier2_value_present(self):
        tbl = _moca_table(oid=[1, 2], parallax=[10.0, None])
        m = eh.compute_source_metrics(tbl, "moca", status="ok")
        assert m["n_matched"] == 2
        assert m["tier2_ok"] is True

    def test_tier2_failed_when_matched_but_all_tier2_empty(self):
        tbl = _moca_table(oid=[1, 2], parallax=[None, None])
        m = eh.compute_source_metrics(tbl, "moca", status="ok")
        assert m["tier2_ok"] is False

    def test_tier2_ok_when_no_tier1_match(self):
        tbl = _moca_table(oid=[None, None], parallax=[None, None])
        m = eh.compute_source_metrics(tbl, "moca", status="ok")
        assert m["tier2_ok"] is True


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

class TestEvaluate:
    def _metrics(self, frac=0.7, n_valid=100, status="ok", **extra):
        m = {"status": status, "n_total": n_valid, "n_valid_ids": n_valid,
             "n_matched": int(round(frac * n_valid)), "frac": frac}
        m.update(extra)
        return m

    def test_healthy_passes(self):
        v = eh.evaluate("gaia", self._metrics(frac=0.7), prev=self._metrics(frac=0.71))
        assert v.passed
        assert v.reasons == []

    def test_below_floor_warns(self):
        v = eh.evaluate("moca", self._metrics(frac=0.2), prev=None)
        assert not v.passed
        assert any("floor" in r for r in v.reasons)

    def test_status_failed_warns(self):
        v = eh.evaluate("gaia", self._metrics(frac=0.0, status="failed"), prev=None)
        assert not v.passed
        assert any("failed" in r for r in v.reasons)

    def test_status_skipped_passes(self):
        v = eh.evaluate("gaia", self._metrics(frac=0.0, status="skipped"), prev=None)
        assert v.passed

    def test_regression_beyond_tolerance_warns(self):
        # 10% tolerance: 0.71 -> 0.60 is a ~15% relative drop.
        v = eh.evaluate("gaia", self._metrics(frac=0.60), prev=self._metrics(frac=0.71))
        assert not v.passed
        assert any("drop" in r or "previous" in r for r in v.reasons)

    def test_small_drop_within_tolerance_passes(self):
        # 0.71 -> 0.69 is a ~3% relative drop, under 10%.
        v = eh.evaluate("gaia", self._metrics(frac=0.69), prev=self._metrics(frac=0.71))
        assert v.passed

    def test_no_prior_skips_regression(self):
        # Would be a huge "drop" but there is no prior frac -> only floor applies.
        v = eh.evaluate("gaia", self._metrics(frac=0.60), prev=None)
        assert v.passed

    def test_legacy_string_prior_skips_regression(self):
        # Legacy provenance stored a bare string, not a metrics dict.
        v = eh.evaluate("gaia", self._metrics(frac=0.60), prev="ok")
        assert v.passed

    def test_valid_id_collapse_warns(self):
        v = eh.evaluate(
            "gaia",
            self._metrics(frac=0.7, n_valid=50),
            prev=self._metrics(frac=0.7, n_valid=100),
        )
        assert not v.passed
        assert any("valid" in r.lower() for r in v.reasons)

    def test_moca_tier2_failure_warns(self):
        v = eh.evaluate("moca", self._metrics(frac=0.7, tier2_ok=False), prev=None)
        assert not v.passed
        assert any("tier" in r.lower() for r in v.reasons)

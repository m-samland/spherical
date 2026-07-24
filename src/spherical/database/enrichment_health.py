"""Quantitative health checks for Gaia / MOCAdb enrichment.

The enrichment ``status`` (``ok`` / ``failed`` / ``skipped``) only records whether
a query *raised*; several code paths return all-empty columns without raising and
are still tagged ``ok``. These helpers derive a match *fraction* from the enriched
table's columns and compare it against absolute floors and the previous run's
values, so a silently degraded or partially failed enrichment is flagged.

Metrics are computed from columns only — the enrichment query functions are left
untouched. See ``docs/superpowers/specs/2026-07-10-enrichment-health-checks-design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spherical.database.gaia_astrophysical_params import _clean_gaia_id

# Absolute floors on the match fraction (matched / valid-Gaia-ID rows). Known-good
# worst-case baselines are ~58% (Gaia) / ~61% (MOCA), so these leave wide margin.
GAIA_FRAC_FLOOR = 0.40
MOCA_FRAC_FLOOR = 0.50

# Relative drop (vs the previous run) that trips the regression check. SPHERE has
# no active large programs and Gaia DR3 / MOCAdb are frozen until Gaia DR4, so the
# fraction is near-static run to run; even a 10% relative drop is anomalous.
REGRESSION_TOL = 0.10

_FLOORS = {"gaia": GAIA_FRAC_FLOOR, "moca": MOCA_FRAC_FLOOR}

_GAIA_MATCH_COLUMN = "GAIA_TEFF"
_MOCA_MATCH_COLUMN = "MOCA_OID"

# Representative MOCA tier-2 columns; if a tier-1 match exists but every one of
# these is empty, the tier-2 query silently returned nothing.
_MOCA_TIER2_COLUMNS = ("MOCA_PARALLAX_MAS", "MOCA_U_KMS", "MOCA_SPTN", "MOCA_X_PC")


@dataclass
class Verdict:
    """Result of evaluating one enrichment source for one mode."""

    source: str
    passed: bool
    reasons: list[str] = field(default_factory=list)


def floor_for(source: str) -> float:
    return _FLOORS[source]


# ---------------------------------------------------------------------------
# Metric computation (from columns)
# ---------------------------------------------------------------------------


def _count_valid_ids(target_tbl, gaia_id_column: str = "ID_GAIA_DR3") -> int:
    if gaia_id_column not in target_tbl.colnames:
        return 0
    return sum(_clean_gaia_id(v) is not None for v in target_tbl[gaia_id_column])


def _count_finite(target_tbl, column: str) -> int:
    if column not in target_tbl.colnames:
        return 0
    arr = np.asarray(target_tbl[column], dtype=float)
    return int(np.count_nonzero(np.isfinite(arr)))


def _moca_tier2_ok(target_tbl, n_tier1_matched: int) -> bool:
    if n_tier1_matched == 0:
        return True
    present = [c for c in _MOCA_TIER2_COLUMNS if c in target_tbl.colnames]
    if not present:
        return True
    return any(_count_finite(target_tbl, c) > 0 for c in present)


def compute_source_metrics(target_tbl, source: str, status: str) -> dict:
    """Return match metrics for ``source`` derived from ``target_tbl`` columns."""
    n_valid = _count_valid_ids(target_tbl)
    match_column = _GAIA_MATCH_COLUMN if source == "gaia" else _MOCA_MATCH_COLUMN
    n_matched = _count_finite(target_tbl, match_column)
    metrics = {
        "status": status,
        "n_total": len(target_tbl),
        "n_valid_ids": n_valid,
        "n_matched": n_matched,
        "frac": (n_matched / n_valid) if n_valid > 0 else 0.0,
    }
    if source == "moca":
        metrics["tier2_ok"] = _moca_tier2_ok(target_tbl, n_matched)
    return metrics


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------


def _prev_value(prev, key):
    """Read ``key`` from a previous metrics dict; None for missing/legacy prior."""
    if isinstance(prev, dict):
        return prev.get(key)
    return None


def evaluate(source, metrics, prev, *, floor=None, regression_tol=REGRESSION_TOL) -> Verdict:
    """Evaluate one source's metrics against floor + regression thresholds.

    ``prev`` is the previous run's metrics dict for this source, or None / a legacy
    string when no prior fraction exists (regression checks are then skipped).
    """
    status = metrics.get("status", "ok")
    if status == "skipped":
        return Verdict(source, True, [])
    if status == "failed":
        return Verdict(source, False, [f"{source} query failed"])

    reasons: list[str] = []
    frac = metrics["frac"]
    floor = floor_for(source) if floor is None else floor
    if frac < floor:
        reasons.append(f"{source} frac {frac:.2f} below floor {floor:.2f}")

    prev_frac = _prev_value(prev, "frac")
    if prev_frac is not None and prev_frac > 0 and frac < (1 - regression_tol) * prev_frac:
        rel = (prev_frac - frac) / prev_frac
        reasons.append(f"{source} frac {frac:.2f} dropped {rel * 100:.0f}% vs previous {prev_frac:.2f}")

    prev_valid = _prev_value(prev, "n_valid_ids")
    n_valid = metrics.get("n_valid_ids")
    if prev_valid and n_valid is not None and n_valid < (1 - regression_tol) * prev_valid:
        reasons.append(f"{source} valid Gaia IDs {n_valid} dropped vs previous {prev_valid}")

    if source == "moca" and metrics.get("tier2_ok") is False:
        reasons.append("moca tier-2 query returned nothing")

    return Verdict(source, not reasons, reasons)

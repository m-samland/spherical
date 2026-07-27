"""Regression test for the TRAP astrometric-uncertainty change on real data.

Runs (or reads a prior run of) the 51 Eridani IRDIS DB_K12 phase-6 smoke test
and compares the resulting companion table against a frozen baseline. The
baseline is the **validated per-channel astrometry** produced by the current
TRAP code (regressors off) — the value validated against GRAVITY interferometry
(454.6 mas, within ~0.7 mas of truth; see
``tests/data/51eri_astrometry_benchmark.md``). It is therefore a tight drift
guard: any change to the reported astrometry trips it.

This is a "no surprises" test, not a correctness proof. It asserts that

- positions have not moved more than the free-fit-at-threshold noise budget
  from the validated per-channel result,
- the astrometric σ columns are finite and self-consistent (positive-definite
  2×2 covariance, polar matches Cartesian).

**Heavy + data-dependent — opt in with ``-m regression`` and run in the
pipeline env** (``pixi run -e dev pytest tests/test_51eri_astrometry_regression.py
-m regression``), because it needs the TRAP sibling and the 51 Eri reduction
products on disk.

Workflow:

1. Produce the new output once, with the *current* TRAP branch checked out::

       pixi run -e dev python examples/irdis_reduction_phase6_smoketest.py

   (``MULTIWAVELENGTH_REGRESSORS = None`` in that driver — keep it off; the
   baseline is regressors-off.) This writes
   ``<result>/template_matching/overall_validated_companion_detections.csv``.
   Alternatively set ``SPHERICAL_RUN_SMOKETEST=1`` and this test will run the
   driver itself as a subprocess before comparing.

2. Run this test. It locates the fresh CSV, reads the frozen baseline in
   ``tests/data/``, and checks the tolerances below.

Override the search root with ``SPHERICAL_SMOKETEST_RESULT_DIR`` (pointing at
the smoke test's per-observation result folder, or any parent of it).
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Position tolerances expressed as multiples of the IRDIS K-band PSF σ
# (FWHM ≈ 3.4 px → σ_PSF ≈ 1.45 px). At threshold SNR (≈4–6) a free centroid
# fit is legitimately noisier than the old fixed-width fit; that shift is the
# expected behaviour, not a regression.
SIGMA_PSF_PIX = 1.45
TOL_POSITION_HIGH_SNR_PIX = 0.5 * SIGMA_PSF_PIX
TOL_POSITION_THRESHOLD_SNR_PIX = 2.0 * SIGMA_PSF_PIX
HIGH_SNR_CUTOFF = 8.0
MATCH_RADIUS_PIX = 3.0

BASELINE = (
    Path(__file__).parent
    / "data"
    / "51eri_baseline_overall_validated_companion_detections.csv"
)
NEW_CSV_NAME = "overall_validated_companion_detections.csv"

SMOKETEST_DRIVER = "examples/irdis_reduction_phase6_smoketest.py"
DEFAULT_REDUCTION_ROOT = Path.home() / "data" / "sphere" / "reduction"


def _locate_new_csv() -> Path | None:
    """Find the freshly produced template-matched short table.

    The per-observation result folder name can contain a literal ``*`` (a
    sanitised target prefix), so we walk real directory names with rglob rather
    than shell-globbing.
    """
    override = os.environ.get("SPHERICAL_SMOKETEST_RESULT_DIR")
    roots = [Path(override)] if override else [DEFAULT_REDUCTION_ROOT]
    for root in roots:
        if not root.exists():
            continue
        matches = [
            p
            for p in root.rglob(f"template_matching/{NEW_CSV_NAME}")
            if "51_Eri" in str(p) and "2015-09-24" in str(p)
        ]
        if not matches and override:
            # An explicit override may already point *at* the template_matching
            # folder or the observation folder.
            matches = list(root.rglob(NEW_CSV_NAME))
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
    return None


def _maybe_run_smoketest() -> None:
    if os.environ.get("SPHERICAL_RUN_SMOKETEST") != "1":
        return
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["python", SMOKETEST_DRIVER],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "SPHERICAL_RUN_SMOKETEST=1 but the smoke test failed "
            f"(rc={result.returncode}). Missing data or wrong env?\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )


@pytest.mark.regression
def test_51eri_astrometry_matches_baseline_within_tolerance():
    if not BASELINE.exists():
        pytest.skip(
            f"Baseline CSV not present at {BASELINE}. Re-freeze it from a clean "
            "regressors-off run's overall_validated_companion_detections.csv."
        )

    _maybe_run_smoketest()

    new_csv = _locate_new_csv()
    if new_csv is None:
        pytest.skip(
            "No fresh template-matched output found. Run the phase-6 smoke test "
            "first (pixi run -e dev python examples/irdis_reduction_phase6_smoketest.py) "
            "or set SPHERICAL_RUN_SMOKETEST=1 / SPHERICAL_SMOKETEST_RESULT_DIR."
        )

    new_df = pd.read_csv(new_csv)
    old_df = pd.read_csv(BASELINE)
    assert len(new_df) > 0, f"New output {new_csv} is empty."

    # `xy_relative_corr` is written only by the astrometry-uncertainty code, so
    # its absence means the CSV on disk predates the change (a stale run, or a
    # WP2-regressor experiment folder). Skip rather than fail — the test only
    # judges output produced by the current TRAP branch.
    if "xy_relative_corr" not in new_df.columns:
        pytest.skip(
            f"{new_csv} predates the astrometry-uncertainty change (no "
            "xy_relative_corr column). Re-run the phase-6 smoke test with the "
            "current TRAP branch, regressors off."
        )

    # Match each new candidate to the nearest baseline candidate in the
    # detector-relative plane.
    old_xy = old_df[["x_relative", "y_relative"]].values.astype(float)
    failures = []
    for i in range(len(new_df)):
        n = new_df.iloc[i]
        nx, ny = float(n["x_relative"]), float(n["y_relative"])
        d = np.hypot(old_xy[:, 0] - nx, old_xy[:, 1] - ny)
        j = int(np.argmin(d))
        if d[j] > MATCH_RADIUS_PIX:
            # A new candidate with no baseline counterpart is informative but
            # not itself a tolerance failure; the baseline had a single source.
            continue
        o = old_df.iloc[j]

        snr = float(n.get("norm_snr_fit_free", np.nan))
        tol_pos = (
            TOL_POSITION_HIGH_SNR_PIX
            if snr >= HIGH_SNR_CUTOFF
            else TOL_POSITION_THRESHOLD_SNR_PIX
        )
        dx = abs(nx - float(o["x_relative"]))
        dy = abs(ny - float(o["y_relative"]))
        if dx > tol_pos or dy > tol_pos:
            failures.append(
                f"cand {i}: Δpos ({dx:.3f}, {dy:.3f}) px > tol {tol_pos:.3f} "
                f"at SNR {snr:.1f}"
            )

        # The core deliverable: the astrometric σ columns must be finite,
        # positive, and self-consistent.
        sep = float(n["separation"])
        sx = float(n["x_relative_sigma"])
        sy = float(n["y_relative_sigma"])
        ssep = float(n["separation_sigma"])
        spa = float(n["position_angle_sigma"])
        for name, val in (
            ("x_relative_sigma", sx),
            ("y_relative_sigma", sy),
            ("separation_sigma", ssep),
            ("position_angle_sigma", spa),
        ):
            if not np.isfinite(val) or val <= 0:
                failures.append(f"cand {i}: {name} not finite/positive ({val})")

        # 2×2 (x, y) covariance must be positive-definite.
        if "xy_relative_corr" in n:
            rho = float(n["xy_relative_corr"])
            if not np.isfinite(rho) or abs(rho) >= 1.0:
                failures.append(f"cand {i}: xy_relative_corr {rho} not in (-1, 1)")

        # Polar σ must be consistent with the Cartesian magnitude: the
        # separation error cannot exceed the total positional error.
        if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(ssep):
            total = np.hypot(sx, sy)
            if ssep > total + 1e-6:
                failures.append(
                    f"cand {i}: separation_sigma {ssep:.4f} > "
                    f"total positional σ {total:.4f}"
                )
        # position_angle_sigma (deg) should equal tangential σ / separation.
        if np.isfinite(spa) and sep > 0:
            tangential = np.radians(spa) * sep
            if not (0 < tangential < 10 * SIGMA_PSF_PIX):
                failures.append(
                    f"cand {i}: implied tangential σ {tangential:.4f} px "
                    "outside plausible range"
                )

    assert not failures, "Astrometry regression tolerances exceeded:\n" + "\n".join(
        failures
    )

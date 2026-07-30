"""Regression test for the 51 Eridani **IFS** astrometry on real data.

The IRDIS sibling (:mod:`tests.test_51eri_astrometry_regression`) guards the
DB_K12 result. This module guards the simultaneous IFS YJH reduction of the same
night, and additionally freezes the *conclusions* reached in
``tests/data/51eri_astrometry_benchmark.md`` §8, so a code change cannot quietly
undo them:

- the reported astrometry comes from the **template collapse**, not the
  per-channel override (only 2 of 37 channels clear ``candidate_threshold``, far
  below ``per_channel_min_channel_fraction``; the override was 4.4 mas *further*
  from GRAVITY),
- the collapse position agrees with the GRAVITY ground truth within its own σ at
  the nominal 7.46 mas/px plate scale, so this dataset does not demand a scale
  revision,
- the σ columns are finite, positive and self-consistent,
- ``n_templates_above_threshold`` matches the per-template tables actually
  written by the run (stale-CSV contamination guard).

**Heavy + data-dependent — opt in with ``-m regression`` and run in the pipeline
env** (``pixi run -e dev pytest tests/test_51eri_ifs_astrometry_regression.py
-m regression``), because it needs the 51 Eri IFS TRAP products on disk.

Produce them with a driver that has ``run_trap_reduction`` / ``run_trap_detection``
enabled, e.g. ``examples/ifs_reduction_template.py`` pointed at 51 Eri, then run
this test. Override the search root with ``SPHERICAL_IFS_RESULT_DIR`` (the
per-observation TRAP result folder, or any parent of it).

GRAVITY values are **unpublished** — do not redistribute outside this project.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# IFS-specific constants. The plate scale is the single most common mistake when
# reusing the IRDIS benchmark — it is 7.46 mas/px here, not 12.25.
IFS_PLATE_SCALE_MAS = 7.46
N_USABLE_CHANNELS = 37
# Mean diffraction FWHM over the 37 used YJH channels is 4.66 px (0.94–1.68 µm,
# 8 m aperture, 7.46 mas/px) → σ_PSF ≈ 1.98 px ≈ 14.8 mas.
SIGMA_PSF_PIX = 1.98

# GRAVITY interferometric ground truth for epoch 2015.73 (unpublished).
GRAVITY_RHO_MAS = 455.364
GRAVITY_RHO_SIGMA_MAS = 0.653
GRAVITY_PA_DEG = 166.839
GRAVITY_PA_SIGMA_DEG = 0.086

# Agreement budget against GRAVITY. Separation is checked in units of the
# combined σ; the PA residual is dominated by the true-north/pupil-offset
# calibration folded into DEROT ANGLE, so it gets an absolute degree budget
# rather than an nσ one.
MAX_SEPARATION_NSIGMA = 2.0
MAX_POSITION_ANGLE_RESIDUAL_DEG = 1.0

# Drift budget against the frozen baseline. At the ≈6σ SNR of this detection a
# free centroid fit moves by a sizeable fraction of σ_PSF between runs; anything
# beyond half a PSF σ is a real change, not fit noise.
TOL_POSITION_PIX = 0.5 * SIGMA_PSF_PIX
MATCH_RADIUS_PIX = 3.0

BASELINE_OVERALL = (
    Path(__file__).parent
    / "data"
    / "51eri_ifs_baseline_overall_validated_companion_detections.csv"
)
BASELINE_PER_CHANNEL = (
    Path(__file__).parent / "data" / "51eri_ifs_baseline_per_channel_astrometry.csv"
)
OVERALL_CSV_NAME = "overall_validated_companion_detections.csv"
PER_CHANNEL_CSV_NAME = "per_channel_astrometry.csv"

DEFAULT_IFS_TRAP_ROOT = Path.home() / "data" / "sphere" / "reduction" / "IFS" / "trap"


def _locate_template_matching_dir() -> Path | None:
    """Find the freshest 51 Eri IFS ``template_matching/`` folder.

    The per-observation folder name can contain a literal ``*`` (a sanitised
    target prefix), so walk real directory names with rglob rather than
    shell-globbing. ``old_reduction`` / ``old_logs`` subfolders hold deliberately
    retained previous runs and must never be picked up.
    """
    override = os.environ.get("SPHERICAL_IFS_RESULT_DIR")
    root = Path(override) if override else DEFAULT_IFS_TRAP_ROOT
    if not root.exists():
        return None
    matches = [
        p.parent
        for p in root.rglob(f"template_matching/{OVERALL_CSV_NAME}")
        if "51_Eri" in str(p) and "2015-09-24" in str(p) and "old_reduction" not in p.parts
    ]
    if not matches and override:
        matches = [p.parent for p in root.rglob(OVERALL_CSV_NAME)]
    if not matches:
        return None
    return sorted(matches, key=lambda p: (p / OVERALL_CSV_NAME).stat().st_mtime)[-1]


def _require_output() -> Path:
    if not BASELINE_OVERALL.exists():
        pytest.skip(
            f"Baseline CSV not present at {BASELINE_OVERALL}. Re-freeze it from a "
            "clean 51 Eri IFS run's overall_validated_companion_detections.csv."
        )
    found = _locate_template_matching_dir()
    if found is None:
        pytest.skip(
            "No 51 Eri IFS template-matched output found. Run an IFS driver with "
            "TRAP enabled first, or set SPHERICAL_IFS_RESULT_DIR."
        )
    return found


@pytest.mark.regression
def test_ifs_astrometry_matches_frozen_baseline():
    """Reported position has not drifted from the frozen IFS baseline."""
    tm_dir = _require_output()
    new_df = pd.read_csv(tm_dir / OVERALL_CSV_NAME)
    old_df = pd.read_csv(BASELINE_OVERALL)
    assert len(new_df) > 0, f"New output {tm_dir / OVERALL_CSV_NAME} is empty."

    old_xy = old_df[["x_relative", "y_relative"]].values.astype(float)
    failures = []
    for i in range(len(new_df)):
        n = new_df.iloc[i]
        nx, ny = float(n["x_relative"]), float(n["y_relative"])
        d = np.hypot(old_xy[:, 0] - nx, old_xy[:, 1] - ny)
        j = int(np.argmin(d))
        if d[j] > MATCH_RADIUS_PIX:
            # A new candidate with no baseline counterpart is informative but not
            # itself a tolerance failure; the baseline had a single source.
            continue
        o = old_df.iloc[j]
        dx = abs(nx - float(o["x_relative"]))
        dy = abs(ny - float(o["y_relative"]))
        if dx > TOL_POSITION_PIX or dy > TOL_POSITION_PIX:
            failures.append(
                f"cand {i}: Δpos ({dx:.3f}, {dy:.3f}) px > tol {TOL_POSITION_PIX:.3f} px"
            )

    assert not failures, "IFS astrometry drifted from baseline:\n" + "\n".join(failures)


@pytest.mark.regression
def test_ifs_astrometry_agrees_with_gravity():
    """The collapse position sits on the GRAVITY truth at 7.46 mas/px."""
    tm_dir = _require_output()
    n = pd.read_csv(tm_dir / OVERALL_CSV_NAME).iloc[0]

    rho = float(n["separation"]) * IFS_PLATE_SCALE_MAS
    rho_sigma = float(n["separation_sigma"]) * IFS_PLATE_SCALE_MAS
    d_rho = rho - GRAVITY_RHO_MAS
    n_sigma = d_rho / np.hypot(rho_sigma, GRAVITY_RHO_SIGMA_MAS)
    assert abs(n_sigma) <= MAX_SEPARATION_NSIGMA, (
        f"separation {rho:.2f} ± {rho_sigma:.2f} mas is {n_sigma:+.2f}σ from "
        f"GRAVITY {GRAVITY_RHO_MAS} ± {GRAVITY_RHO_SIGMA_MAS} mas "
        f"(Δ = {d_rho:+.2f} mas at {IFS_PLATE_SCALE_MAS} mas/px)"
    )

    d_pa = float(n["position_angle"]) - GRAVITY_PA_DEG
    assert abs(d_pa) <= MAX_POSITION_ANGLE_RESIDUAL_DEG, (
        f"position angle {n['position_angle']:.3f}° is {d_pa:+.3f}° from GRAVITY "
        f"{GRAVITY_PA_DEG}° (true north already applied in DEROT ANGLE)"
    )


@pytest.mark.regression
def test_ifs_reports_collapse_not_per_channel_astrometry():
    """The per-channel override stays gated off for IFS.

    Only 2 of 37 channels clear ``candidate_threshold`` here, so the override
    would report a position built from 5% of the data — 4.4 mas further from
    GRAVITY than the collapse, with a *smaller* σ because the two adjacent
    H-band channels are speckle-correlated but combined as independent. The gate
    (``per_channel_min_channel_fraction``, default 0.5) must keep it off.
    """
    tm_dir = _require_output()
    n = pd.read_csv(tm_dir / OVERALL_CSV_NAME).iloc[0]
    assert n["astrometry_source"] == "collapse", (
        f"astrometry_source is {n['astrometry_source']!r}; the per-channel "
        "override must remain gated off for IFS (see benchmark §8c)"
    )

    per_channel_csv = tm_dir / PER_CHANNEL_CSV_NAME
    assert per_channel_csv.exists(), (
        f"{PER_CHANNEL_CSV_NAME} missing — the per-channel diagnostic must still "
        "be written even when the override is gated off"
    )
    p = pd.read_csv(per_channel_csv).iloc[0]
    n_channels = float(p["channels_above_threshold"])
    assert n_channels < 0.5 * N_USABLE_CHANNELS, (
        f"{n_channels:.0f} of {N_USABLE_CHANNELS} channels now clear the candidate "
        "threshold — the gate would flip to per-channel astrometry. Re-validate "
        "against GRAVITY before re-freezing the baseline."
    )


@pytest.mark.regression
def test_ifs_astrometric_sigmas_are_finite_and_self_consistent():
    """σ columns are finite, positive, and mutually consistent."""
    tm_dir = _require_output()
    n = pd.read_csv(tm_dir / OVERALL_CSV_NAME).iloc[0]

    sep = float(n["separation"])
    sx = float(n["x_relative_sigma"])
    sy = float(n["y_relative_sigma"])
    ssep = float(n["separation_sigma"])
    spa = float(n["position_angle_sigma"])

    failures = []
    for name, val in (
        ("x_relative_sigma", sx),
        ("y_relative_sigma", sy),
        ("separation_sigma", ssep),
        ("position_angle_sigma", spa),
    ):
        if not np.isfinite(val) or val <= 0:
            failures.append(f"{name} not finite/positive ({val})")

    rho = float(n["xy_relative_corr"])
    if not np.isfinite(rho) or abs(rho) >= 1.0:
        failures.append(f"xy_relative_corr {rho} not in (-1, 1)")

    if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(ssep):
        total = np.hypot(sx, sy)
        if ssep > total + 1e-6:
            failures.append(
                f"separation_sigma {ssep:.4f} > total positional σ {total:.4f}"
            )

    # A sub-mas σ on a ~6σ detection would mean the error budget collapsed; the
    # benchmark's plausible range is 1–4 mas (0.13–0.54 px).
    if np.isfinite(ssep):
        ssep_mas = ssep * IFS_PLATE_SCALE_MAS
        if not (1.0 <= ssep_mas <= 6.0):
            failures.append(f"separation_sigma {ssep_mas:.2f} mas outside 1–6 mas")

    if np.isfinite(spa) and sep > 0:
        tangential = np.radians(spa) * sep
        if not (0 < tangential < 10 * SIGMA_PSF_PIX):
            failures.append(
                f"implied tangential σ {tangential:.4f} px outside plausible range"
            )

    assert not failures, "IFS astrometric σ inconsistent:\n" + "\n".join(failures)


@pytest.mark.regression
def test_ifs_no_stale_template_contamination():
    """``n_templates_above_threshold`` matches the tables this run wrote.

    trap purges the per-template and overall companion tables before writing, so
    a template that found nothing cannot leave last run's file behind. This
    asserts the two views agree.
    """
    tm_dir = _require_output()
    n = pd.read_csv(tm_dir / OVERALL_CSV_NAME).iloc[0]
    written = sorted(p.name for p in tm_dir.glob("validated_companion_table_[!s]*.csv"))
    assert len(written) == int(n["n_templates_above_threshold"]), (
        f"n_templates_above_threshold = {n['n_templates_above_threshold']} but "
        f"{len(written)} per-template validated tables on disk: {written}"
    )
    assert f"validated_companion_table_{n['template_name']}.csv" in written, (
        f"reported template {n['template_name']!r} has no per-template table in {written}"
    )

# 51 Eridani b — astrometry benchmark & baseline reference

Reference document for evaluating pipeline astrometry (position + uncertainties) of the
51 Eri b companion against external ground truth. Written 2026-07-27 from the IRDIS
DB_K12 phase-6 smoke test; **designed to be reused for the simultaneous IFS dataset**.

> **History (2026-07-27), read this first:** this benchmark went through two corrections.
> (1) An early draft compared a `flat` "K1-only" detection to GRAVITY using a *stale* CSV
> that the pipeline erroneously ingested — a real bug, now fixed
> (`../trap/docs/llm_reference/GITHUB_ISSUE_stale_template_csv_ingestion.md`).
> (2) After fixing that, a **clean, freshly-computed per-channel measurement** confirmed the
> underlying point anyway: the per-channel astrometry (channels above threshold, here K1
> only) lands on GRAVITY (Δρ ≈ −0.7 mas) while the template collapse is +8.7 mas / 2.6σ
> off. TRAP now **measures per-channel astrometry and reports it as primary**
> (`astrometry_source` column), collapse as fallback. The numbers below reflect that.

All conversions use the instrument plate scale; see the provenance table for exact values.

---

## 1. Dataset identification

| Property | Value |
|---|---|
| Target | 51 Eridani (companion 51 Eri b) |
| Night (`NIGHT_START`) | 2015-09-24 |
| Epoch | ≈ 2015.73 (Maire et al. 2019 lists this observation as **2015.74**) |
| Instrument / mode | VLT/SPHERE **IRDIS** DBI, filter **DB_K12** |
| Channels | ch0 = K1 (2.110 µm), ch1 = K2 (2.251 µm) |
| Derotator | PUPIL (ADI) |
| IRDIS plate scale | **12.25 mas/px** (`0.01225 arcsec/px`, `trap_config_for_irdis()`) |
| Anamorphism | `yx_anamorphism = [1.0062, 1.0]` |
| TRAP search annulus | inner 31 px, outer 43 px; `yx_known_companion_position = (-35.95, -8.43)` |
| `temporal_components_fraction` | `[0.2]` |
| Multiwavelength regressors | **None** (WP1 baseline; regressors OFF — the baseline condition) |
| Driver | `examples/irdis_reduction_phase6_smoketest.py` (full) or a detection-only re-run |
| σ_PSF (K-band) | FWHM ≈ 3.4 px → σ_PSF ≈ **1.44 px ≈ 17.7 mas** |

> **Regressors must be OFF for the baseline.** Use `template_matching/` (regressors off)
> or the frozen `template_matching_without_sdi/`. The `template_matching_{pool,occluded,
> sdi}/` folders are WP2 regressor experiments and must **not** be used as an astrometry
> baseline.
>
> **Clear stale per-template CSVs before re-running detection.** A template that finds
> nothing in a run leaves its previous `companion_table_*.csv` on disk. The combination
> bug that ingested those has been fixed (it now uses in-memory tables), but stale files
> on disk are still confusing when inspecting output by hand.

---

## 2. How template matching works here (important for interpretation)

TRAP template matching does **not** detect per channel and combine. For each spectral
template it **collapses** all wavelength channels into a *single* template-weighted
detection map, then detects on that one map (`template_matching_detection` →
`wavelength_indices=[0]`, `detection.py:3857`). Consequences:

- `channels_above_threshold = 1` for **every** template — the templates differ only in the
  spectral weighting of the collapse, not in per-channel inclusion.
- The per-detection σ is a **single 2D-Gaussian-fit σ on the collapsed map** (LevMar
  param-cov + Cramér-Rao floor), not a combination of independent per-channel σ.
- Multiple templates (flat, L-type, T-type, …) each yield a position; the cross-template
  step picks the highest-SNR template and reports the across-template scatter as a
  **diagnostic flag only** (never combined into σ — templates re-weight the same data, so
  their scatter is not an independent error).

**Per-channel astrometry is now measured and is primary (2026-07-27).** Because the
collapse is optimized for detection SNR, not astrometry — it folds in signal-free channels
whose speckle structure biases the centroid — TRAP now also runs a **template-independent
per-channel detection** (`measure_per_channel_astrometry`): it detects and fits the source
in each wavelength channel and combines the channels that individually clear the detection
threshold via `_combine_channels_rt_frame` (source-aligned inverse-variance). The reported
`overall_*` position/σ is taken from this per-channel combination when a source is detected
in individual channels (`astrometry_source = "per_channel"`), falling back to the collapse
otherwise (`astrometry_source = "collapse"`). Detection significance (`norm_snr_fit_free`,
`best_template`) and the spectrum still come from the template collapse. A standalone
`per_channel_astrometry.csv` is written alongside the per-template tables.

---

## 3. Ground truth — GRAVITY interferometry (**unpublished**)

Best available truth for this epoch (~5–8× more precise than any SPHERE imaging value).

| Quantity | Value |
|---|---|
| RA offset | 103.751 ± 0.698 mas |
| Dec offset | −443.382 ± 0.636 mas |
| **Separation** | **455.364 ± 0.653 mas** |
| **Position angle** | **166.839 ± 0.086°** |

Implied separation in detector pixels: 455.364 / 12.25 = **37.17 px** (IRDIS);
455.364 / 7.46 = **61.04 px** (IFS — see §7).

---

## 4. Published reference — Maire et al. 2019, Table A.1

1σ, measurement-procedure-only uncertainties. Only the **2015.74** rows correspond to this
dataset; later epochs are shown for context (they trace the orbital motion in PA).

**SDI + TLOCI**

| Epoch | Band | ρ (mas) | PA (°) |
|---|---|---|---|
| **2015.74** | **K1** | **453.4 ± 4.4** | **167.15 ± 0.55** |
| 2015.74 | H | 453.9 ± 15.9 | 166.1 ± 2.0 |
| 2016.04 | H2 | 456.7 ± 6.6 | 165.50 ± 0.83 |
| 2016.95 | H2 | 453.6 ± 5.7 | 160.30 ± 0.72 |
| 2017.74 | K1 | 449.0 ± 2.9 | 155.67 ± 0.37 |
| 2018.72 | K1 | 443.3 ± 4.2 | 150.23 ± 0.54 |

**SDI + ANDROMEDA**

| Epoch | Band | ρ (mas) | PA (°) |
|---|---|---|---|
| **2015.74** | **K1** | **448.6 ± 1.4** | **167.45 ± 0.06** |
| 2015.74 | H | 467.4 ± 2.9 | 167.09 ± 0.07 |
| 2016.04 | H2 | — (a) | — (a) |
| 2016.95 | H2 | 456.1 ± 1.6 | 160.06 ± 0.06 |
| 2017.74 | K1 | 447.9 ± 1.3 | 155.80 ± 0.04 |
| 2018.72 | K1 | 439.0 ± 1.2 | 150.09 ± 0.03 |

(a) Astrometry could not be extracted at that epoch.

Relevant comparison band is **K1** (this is DB_K12; the T-type collapse is K1-weighted).

---

## 5. Pipeline result — this reduction (clean single run)

Source: `…/IRDIS/trap/*_51_Eri/DB_K12/2015-09-24/template_matching/overall_validated_companion_detections.csv`.
Only the **T-type** template validated; `n_templates_above_threshold = 1`,
`astrometry_template_disagreement = False`. Reported astrometry is **per-channel**
(`astrometry_source = per_channel`; K1 was the only channel above threshold).

### Reported (per-channel primary) — what the overall table now contains

| Column | px | mas / deg |
|---|---|---|
| x_relative / y_relative | −8.552715 / −36.114807 | — |
| **separation** | 37.113720 | **454.64 mas** |
| **separation_sigma** = radial_sigma_stat | 0.469441 | **5.75 mas** |
| tangential_sigma_stat | 0.199621 | 2.45 mas |
| **position_angle** | — | **166.677°** |
| position_angle_sigma | — | **0.308°** |
| astrometry_source | — | per_channel |
| norm_snr_fit_free / best_template | — | 6.420 / T-type (from collapse) |

### For contrast: the template-collapse position (now demoted to fallback)

separation 37.884269 px = **464.08 mas**, PA **165.995°**, radial σ 0.278 px = 3.40 mas.
This is what was reported before the per-channel change; it sits +8.7 mas / 2.6σ from
GRAVITY because the collapse folds in the signal-free K2 channel.

### For reference: OLD frozen baseline (old code, regressors off, no σ)

`tests/data/51eri_baseline_overall_validated_companion_detections.csv` (from
`template_matching_without_sdi/`): separation 37.74646 px = **462.39 mas**,
position_angle **166.152°**, all σ columns **NaN** (the old code produced no
uncertainties).

---

## 6. Comparison to ground truth

| Source | Separation | Δρ vs GRAVITY | PA | ΔPA vs GRAVITY |
|---|---|---|---|---|
| **GRAVITY (truth)** | 455.36 ± 0.65 | — | 166.839 ± 0.086 | — |
| Maire TLOCI (K1) | 453.4 ± 4.4 | −1.96 | 167.15 ± 0.55 | +0.31 |
| Maire ANDROMEDA (K1) | 448.6 ± 1.4 | −6.76 | 167.45 ± 0.06 | +0.61 |
| **Pipeline per-channel (reported)** | 454.64 ± 5.75 | **−0.72** | 166.677 ± 0.31 | **−0.16** |
| Pipeline template-collapse (fallback) | 464.08 ± 3.40 | +8.72 | 165.995 ± 0.34 | −0.84 |
| Old pipeline (no σ) | 462.39 | +7.03 | 166.152 | −0.69 |

The **per-channel** reported value is consistent with GRAVITY (**Δρ −0.72 mas ≈ 0.1σ**,
ΔPA −0.16°); the collapse is 2.6σ / 2.4σ off. This is the same conclusion the stale-file
episode pointed at, now established on freshly-computed clean data.

## 7. Interpretation & caveats

- **Per-channel beats collapse here because the collapse is SNR-optimal, not
  astrometry-optimal.** It folds the signal-free K2 channel (below `candidate_threshold`)
  into the map with a spectral weight, and K2's speckle structure pulls the centroid ~0.9 px
  (~11 mas) outward. Per-channel uses only channels individually above threshold — here K1 —
  so it is unbiased but has a larger formal σ (5.75 vs 3.40 mas radial). Unbiased-but-noisier
  is the right trade for astrometry.
- **n = 1 caveat.** One dataset, one GRAVITY point. The *mechanism* is principled and
  dataset-independent; keep validating the per-channel-vs-collapse gap across more
  companions with interferometric truth (IRDIS and IFS).
- **True north *is* already applied — corrected 2026-07-28.** An earlier version of this
  document claimed the reported PA was uncalibrated. It is not:
  `spherical.database.metadata.compute_angles(true_north=-1.75)` folds true north, the
  pupil offset and the per-instrument offset (`IFS = -100.48`, `IRDIS = 0.0`) into
  `DEROT ANGLE`, and `{coro,center}_parallactic_angles.fits` — what TRAP receives — is
  exactly that column. PA may therefore be compared to GRAVITY directly. The residual
  calibration uncertainty is the RSS of the constants: ≈ 0.14° for IRDIS
  (TN ±0.08, PUPOFF ±0.11) and ≈ 0.17° for IFS (plus the ±0.10 instrument offset).
  What remains genuinely uncalibrated is the **epoch plate scale**.
- **The σ magnitude is realistic.** The per-channel σ_ρ ≈ 5.75 mas and σ_PA ≈ 0.31° are in
  the range of published SPHERE measurement errors for this epoch/band (ANDROMEDA 1.4 mas /
  0.06°; TLOCI 4.4 mas / 0.55°). Radial σ (5.75) > tangential σ (2.45) as expected for ADI
  self-subtraction.
- **The regression test asserts stability + σ self-consistency, not absolute truth.**
  `tests/test_51eri_astrometry_regression.py` checks position drift vs the frozen baseline
  and that σ columns are finite/positive/consistent — *not* agreement with GRAVITY —
  precisely because of the uncalibrated offsets. Tolerances: high-SNR (≥ 8) 0.5 σ_PSF =
  0.725 px; threshold-SNR 2.0 σ_PSF = 2.90 px; match radius 3.0 px; σ_PSF = 1.45 px.
- **Only one template validates here.** flat and L-type produce contrast tables but no
  validated companion; there is no cross-template disagreement to report. (If you see
  disagreement/scatter on this dataset, suspect a stale per-template CSV on disk.)

---

## 8. Evaluating a future IFS run

SPHERE observes IFS and IRDIS **simultaneously** (IRDIFS), so the same night 2015-09-24 has
an IFS cube for 51 Eri b, and **the GRAVITY ground truth in §3 applies unchanged**.

### 8a. IFS-specific constants (differ from IRDIS!)

| Property | IFS value |
|---|---|
| IFS plate scale | **7.46 mas/px** (`0.00746 arcsec/px`) — **not** 12.25 |
| Channels | 39 (YJH ~0.95–1.65 µm); **first and last usually excluded → ~37 usable** |
| GRAVITY separation in IFS px | 455.364 / 7.46 = **61.04 px** |
| Same GRAVITY truth | ρ 455.364 ± 0.653 mas, PA 166.839 ± 0.086° |

**Convert every IFS pixel value with 7.46, not 12.25.** This is the most common mistake to
guard against when reusing this document.

### 8b. Acceptance criteria

1. **Separation:** IFS ρ (px × 7.46) within a few mas of 455.364 mas raw, consistent within
   σ once true-north / plate-scale calibration is applied.
2. **PA:** within ~1° of 166.839° raw (TN offset dominates); consistent within σ after
   calibration.
3. **σ present, finite, and realistic.** The per-detection σ is a single-fit σ on the
   template-collapsed map. Sanity-check it against the collapsed-map noise normalization
   and against GRAVITY; distrust a sub-mas σ.
4. **Cross-template diagnostics are flags only.** With a real spectral library, expect
   `astrometry_template_disagreement` to fire more often for IFS. It must remain a
   diagnostic — `*_sigma_template_scatter` must never be folded into the reported σ
   (templates re-weight the same collapsed data; the scatter is not independent).
5. **No stale contamination.** Confirm `n_templates_above_threshold` equals the number of
   templates that actually validated this run (the stale-CSV bug is fixed, but verify).

### 8b-bis. Results of the first IFS run (2026-07-28) — read before reusing §8b

The IFS end-to-end run happened; §8b's criteria were applied and three of them needed
qualification. Full numbers in [`../../llm_docs/decisions.md`](../../llm_docs/decisions.md)
2026-07-28.

| Measurement | ρ @ 7.46 mas/px, anamorphism applied | Δρ vs GRAVITY | PA | ΔPA |
|---|---|---|---|---|
| T-type collapse (now reported) | 454.30 ± 3.85 | −1.07 (0.3σ) | 166.160 ± 0.292 | −0.68 |
| L-type collapse | 451.99 ± 5.00 | −3.38 | 167.565 ± 0.414 | +0.73 |
| per-channel (2 of 37 ch, now gated off) | 449.86 ± 2.77 | −5.50 | 167.160 ± 0.259 | +0.32 |

Three things the IRDIS benchmark did not anticipate:

1. **`yx_anamorphism` was `[1, 1]` for IFS.** Now `[1.0059, 1.0011]` in
   `trap_config_for_ifs()`. Worth +0.30 px / +2.2 mas here (the naive +2.5 mas is diluted
   to +2.2 by 43° of field rotation, since TRAP applies the distortion per frame).
2. **The per-channel override made IFS astrometry worse, not better** — see §8c.
3. **The plate scale is the open item, not the detection.** Once (1) and (2) are handled the
   T-type collapse sits 1.07 mas (0.3σ) from GRAVITY at the nominal 7.46 mas/px, so this
   dataset does *not* demand a scale revision. Two independent transfers nonetheless
   suggest the charis resampled grid is ~0.5–1% coarser than 7.46: the DM waffle spots
   (fixed `N·λ/D`, identical for both simultaneous subsystems) transfer the IRDIS scale as
   **7.512 mas/px**, and 51 Eri b itself as 7.539. The waffle estimator is robust to the
   anamorphism — the median over a square's side/diagonal pairs averages the orientation
   dependence out to first order regardless of field angle — but not to other optical
   systematics. Settling it needs a real astrometric field. **Always state which scale a
   published IFS separation uses**; 7.46 is the ESO DRH cube convention, and the charis
   extraction builds its own square grid (pitch = 1/√3 lenslet units, 348² cropped to 262²)
   whose pitch has never been calibrated.

### 8c. What differs from IRDIS

- **The single-bad-channel failure mode does not exist** — detection is on one collapsed
  map either way, and IFS collapses ~37 channels, so no individual channel dominates.
  **This prediction was right, and the per-channel override was applied to IFS anyway
  (fixed 2026-07-28).** With `candidate_threshold = 4.75` only channels 31 and 32 clear it
  (per-channel peak norm-SNR at the source: 6.30, 5.46, then 4.57 / 4.56 / 4.49 / 4.17…), so
  the override reported a position built from 2 of 37 channels — discarding the entire
  J-band peak — selected by the very noise that promoted those two, and 4.4 mas further
  from GRAVITY than the collapse. Its σ was also *smaller* than the collapse's (0.371 vs
  0.515 px) while using 5% of the data, because the two adjacent H-band channels are
  speckle-correlated but were combined as independent (χ²_red,radial = 0.034 on 1 dof is
  the tell). trap now gates the override on
  `DetectionParameters.per_channel_min_channel_fraction` (default 0.5) and floors the
  combined σ at the best contributing channel.
- **The collapse extracts no multiplex gain here, which is a separate open question.**
  The quadrature sum of the 37 per-channel norm-SNRs is 16.8; the T-type collapse reaches
  6.19, *below* its best single channel (6.30). Consistent with strongly correlated residual
  speckle across channels — and the same fact that invalidates independent-channel σ
  combination. If that correlation could be whitened this detection should be ~2× more
  significant.
- **The inter-channel-independence concern applies only to the non-template per-wavelength
  path** (`_combine_channels_rt_frame`, n>1), which the production template-matched
  reduction does not use. If that path is ever used for IFS astrometry, its
  independent-channel σ would be over-optimistic for speckle-correlated channels —
  calibrate with the empirical `√χ²_red` scaling (already applied) or a channel jackknife.

### 8d. Quick expected-number table for the IFS test

| Quantity | Expected (from GRAVITY) | Tolerance guidance |
|---|---|---|
| Separation | 455.36 mas = **61.04 IFS px** | within ~1 px raw; within σ after calibration |
| PA | 166.839° | within ~1° raw (TN-dominated) |
| σ_ρ | — | plausibly 1–4 mas; distrust if ≪ 1 mas |
| σ_PA | — | plausibly 0.05–0.5° |

---

## 9. Provenance & reproduction

| Item | Value |
|---|---|
| spherical branch | `feature/trap-astrometry-uncertainty-regression` |
| trap branch (implementation) | `feature/astrometry-uncertainties` |
| Frozen baseline (old code) | `template_matching_without_sdi/` (regressors off) |
| Fresh output (new code) | `template_matching/` (regressors off) |
| Regression test | `tests/test_51eri_astrometry_regression.py` (`-m regression`) |
| Related trap fix | `../trap/docs/llm_reference/GITHUB_ISSUE_stale_template_csv_ingestion.md` |
| Decision entry | `llm_docs/decisions.md` (2026-07-27) |

Reproduce (needs the pipeline extra + trap sibling + data on disk):

```
# Full (reduction + detection):
pixi run -e dev python examples/irdis_reduction_phase6_smoketest.py
# Detection only (reuses reduction products; clear stale per-template CSVs first):
#   rm template_matching/{companion_table_*,validated_companion_table*,overall_*}.csv
#   then run a driver with run_trap_reduction=False, force={"run_trap_detection"}.
pixi run -e dev pytest tests/test_51eri_astrometry_regression.py -m regression -v
```

Conversions used throughout: **IRDIS 12.25 mas/px, IFS 7.46 mas/px**, σ_PSF(K) ≈ 1.44 px.
GRAVITY values are **unpublished** — do not redistribute outside this project.

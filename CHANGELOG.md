# Changelog

All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

### 🐛 Fixed
- **Target names hidden inside pipe-joined `ID_HD` values now resolve locally** – SIMBAD returns
  several designations for one object separated by `|`, and `target_table.extract_ids` preserves
  that (e.g. `ID_HD = "HD 135344|HD 135344A"`). `_build_normalized_id_lookup` indexed the joined
  cell verbatim, so neither designation was individually addressable. 21 HD names across 114
  observation rows were unreachable — `HD 48915` (Sirius), `HD 36705` (AB Dor), `HD 104237`
  (`MAIN_ID = "V* DX Cha"`), `HD 113791` (`MAIN_ID = "* ksi02 Cen"`) among them — and fell
  through to a SIMBAD network query, which is slow and fails offline. Each designation is now
  indexed separately. Empty ID cells are also skipped: masked IDs stringify to `""`, so a blank
  or whitespace-only target name previously matched 4901 of 6094 IRDIS rows instead of nothing.
  No name that already resolved changed its result ([@m-samland](https://github.com/m-samland)).
- **The waffle-spot fit always fits its background pedestal** – Defaulting to always fit Gaussian
  plus offset for the satellite spots. This removes branching behaviour based on the availability
  of CORO files (e.g., when removing the closest CORO frames from a center file to remove speckle halo).
  Pipeline performance and centering position remains unchanged
  ([#129](https://github.com/m-samland/spherical/issues/129), reported by
  [@tomasstolker](https://github.com/tomasstolker)).
- **`minimum_candidate_separation` and its siblings no longer raise `TypeError`** – The
  candidate-search knobs both reduction templates document reached TRAP's
  `DetectionParameters` in `v2.0.1`, but 3.0.0 pinned `v2.0.0`, so uncommenting one raised
  `TypeError` out of `trap_config.detection.merge()`. The `hasattr` guard in
  `run_trap._candidate_search_kwargs()` covers the pipeline reading these fields, not a
  template setting them (reported by [@tomasstolker](https://github.com/tomasstolker),
  [@m-samland](https://github.com/m-samland)).
- **Relative directories in `DirectoryConfig` no longer scatter TRAP outputs** – `base_path`,
  `raw_directory` and `reduction_directory` are expanded and anchored to the current working
  directory whenever they are set, including the post-construction assignment both reduction
  templates use (`config.directories.base_path = ...`), and `species_database_directory` is
  absolutized before it reaches TRAP.
  TRAP's `add_default_templates()` chdirs into the species database directory without
  restoring the cwd ([m-samland/trap#39](https://github.com/m-samland/trap/issues/39)), so a
  relative reduction directory sent the whole `template_matching/` tree under the species
  directory while the run reported success.
  Absolute paths, the documented setup in both reduction templates, were never affected
  ([@m-samland](https://github.com/m-samland)).

### 🔧 Changed
- **The `trap` dependency tracks `main` instead of a tag** – `pyproject.toml` and `pixi.toml`
  point at `trap@main`, so TRAP fixes arrive without a spherical release;
  `pip install --upgrade -e .` refetches it, while pixi locks the commit and needs
  `pixi update trap`. Compatibility now rests solely on `run_trap._MIN_TRAP_VERSION`, raised
  to `2.0.1`; it may only ever name a *released* tag, since `setuptools_scm` reports an
  untagged commit as `2.0.2.devN` — below `2.0.2`
  ([@m-samland](https://github.com/m-samland)).

---

## [3.0.0] - 2026-08-11 – End-to-End IRDIS Reduction

Major release. IRDIS dual-band imaging is now reduced end to end, mirroring the
IFS workflow, and the database ships with an automated build/update workflow.
Breaking: the `overwrite_*` step flags are replaced by `force`, `trap >= 2.0.0`
is required, database table filenames now name the mode instead of encoding a
boolean, and the monitoring scripts changed their column and flag names.

### ✨ Added
- **`$SPHERICAL_DATABASE_DIR` names the table directory once for the whole install** – The
  variable supplies the database directory to `spherical-sync-tables` and
  `spherical-update-database` (whose `--dest` is therefore optional), to `plot_trap_mosaics`,
  and to both reduction templates. An explicit flag or argument always wins. The precedence
  lives in `spherical.database.paths.resolve_database_dir()`, which is stdlib-only and so
  usable from a base install ([@m-samland](https://github.com/m-samland)).
- **Candidate-search knobs forwarded to TRAP** – `minimum_candidate_separation`,
  `candidate_exclusion_radius` and `max_candidates` from `trap_config.detection` now reach
  `detection_and_characterization_with_template_matching()`, and both reduction templates
  document them. They matter because `search_region_inner_bound` must stay small — the inner
  pixels feed the annulus statistics — while the *detection* floor should not, so a residual
  a pixel from the star was being promoted to a candidate and crashing the target. Forwarded
  only when the installed TRAP defines them ([@m-samland](https://github.com/m-samland)).
- **End-to-end IRDIS dual-band imaging reduction** – IRDIS DBI observations now run the
  full chain — download, master calibrations (background, DIT-resolved flat, bad-pixel
  map), preprocessing into `converted/` cubes with analytic inverse variance, waffle-spot
  centering, flux-PSF and spot photometry calibration, and TRAP post-processing with
  spectral template matching — through the same `execute_targets()` entry point and the
  same resume/force semantics as IFS. Configuration lives in `IRDISReductionConfig`
  (with `IRDISCalibrationConfig` / `IRDISPreprocessConfig`); `examples/irdis_reduction_template.py`
  is the reference driver. Star centers for dithered, non-continuous-waffle sequences are
  propagated from the CENTER waffle fits through the DMS header offsets using a global
  anchor averaged over all CENTER frames. Modes without a dual-band filter pair
  (broadband, narrow-band, dual-polarization) fall back to TRAP's contrast-curve and
  candidate-extraction path instead of template matching. Validated end to end against
  the published 51 Eridani b photometry and astrometry
  ([@m-samland](https://github.com/m-samland)).
- **Bad-pixel masks derived from the inverse-variance cube** – New
  `pipeline.ivar_badpixels.bad_pixel_mask_from_ivar()` flags spaxels from the extracted
  inverse-variance cube and supplies TRAP's bad-pixel mask when no calibration
  `badpixel_map.fits` exists (`derive_trap_bad_pixels_from_ivar`, default `True`). On IFS
  the test is exact `ivar == 0` gated by the data footprint
  (`ivar_bad_pixel_ratio_threshold = 0.0`); IRDIS additionally uses a local-median
  threshold (`0.2`). Because TRAP's mask is 2-D per wavelength, per-frame flags are
  collapsed with `ivar_bad_pixel_frame_fraction` (default `0.5`), which keeps transient
  cosmic-ray flags from erasing the regressor pool
  ([@m-samland](https://github.com/m-samland)).
- **Bad-pixel-aware flux-PSF calibration and PSF-core repair** – The flux-PSF step now
  masks bad pixels in the Gaussian centering fit and in the aperture photometry, and
  repairs bad pixels inside the PSF core (radius `1.22 λ/D`) with a per-channel 2-D
  Moffat fit before the subpixel shift, downweighting the repaired inverse variance so
  later consumers can tell model values from data. Falls back to neighbour interpolation
  with a warning when the Moffat residual is poor, and writes the unrepaired PSF cube
  alongside the repaired one for auditing ([@m-samland](https://github.com/m-samland)).
- **Opt-in TRAP inputs: inverse-variance cube and waffle amplitude modulation** –
  `pass_inverse_variance_to_trap` (default `True`) hands TRAP the measured inverse
  variance; `pass_amplitude_modulation_to_trap` (default `False`, continuous-waffle only)
  hands it the satellite-spot amplitude variation
  ([@m-samland](https://github.com/m-samland)).
- **`pass_center_outliers_as_bad_frames_to_trap`** – When set on a continuous-waffle
  observation, frames whose waffle fit was flagged as a temporal outlier are excluded
  from TRAP's reference basis. Ignored with an INFO log for non-waffle observations,
  where a CENTER frame index has no CORO counterpart
  ([@m-samland](https://github.com/m-samland)).
- **Default coronagraph transmission** – TRAP runs now default
  `coronagraph_transmission` to the packaged `N_ALC_JYH_S` curve for the instrument (the
  IFS YJ curve for `OBS_YJ`/`OBS_H`, the IRDIS H23 curve for IRDIS) so contrasts close to
  the coronagraph are not underestimated. Toggle with `apply_coronagraph_transmission`
  (both configs, default `True`); an explicit table set on the trap config always wins
  ([@m-samland](https://github.com/m-samland)).
- **Per-target stellar parameters for TRAP template matching** – The host star's
  effective temperature is now resolved per observation: Gaia DR3, then a spectral-type
  estimate from `SP_TYPE` via a vendored Mamajek (2022) table, then the configured
  default; `log g` comes from Gaia when available. `[Fe/H]` deliberately stays at the
  configured solar value, because TRAP's template grid is solar-only. Controlled by
  `use_gaia_stellar_parameters` (both configs, default `True`); the lookup lives
  entirely in `spherical`, so `trap` stays Gaia-agnostic
  ([@m-samland](https://github.com/m-samland)) ([#109](https://github.com/m-samland/spherical/issues/109)).
- **Automated database build and update workflow** – Two console commands cover getting an
  up-to-date database: `spherical-sync-tables` downloads the latest pre-built tables from
  Zenodo (md5-verified, resumable, `--dry-run`/`--list`) — file and observation tables for
  the selected instrument, the matching target tables and `database_provenance.json` when
  the record offers them, with `--include-polarimetry` and `--include-sam` for those modes;
  a record that already ships provenance keeps it instead of having it reconstructed from
  the file tables. `spherical-update-database`
  extends the file table from the last recorded coverage date to today against the ESO
  archive, rebuilds the target and observation tables for every mode — including the new
  sparse-aperture-masking tables (`ifs_sam`, `irdis_sam`) — and runs Gaia DR3 and MOCAdb
  enrichment. `--enrich-only` re-runs enrichment alone, optionally for a single `--mode`.
  Library equivalents live in `spherical.database.build` and are demonstrated in
  `examples/generate_database.py` ([@m-samland](https://github.com/m-samland)).
- **Database provenance tracking** – Every build records its provenance in
  `database_provenance.json` and in compact `SPH*` FITS header keywords: spherical
  version, data source, ESO query date and coverage range, Gaia data release, enrichment
  status per source, and the build parameters used. Table sets are now self-describing
  and ready to publish ([@m-samland](https://github.com/m-samland)).
- **Quantitative enrichment health checks** – `spherical-update-database` now records a
  per-source Gaia/MOCA match *fraction* and compares it against absolute floors and the
  previous run, so a query that succeeds but returns far fewer matches than a known-good
  run is caught. The CLI prints a per-mode health summary and exits non-zero when an
  enrichment failed or dropped more than 10% relative to the prior run; thresholds live
  in `spherical.database.enrichment_health`. Infrastructure failures raise dedicated
  exceptions and are recorded as `failed` before the table is modified, so a transient
  outage cannot silently overwrite good columns with empty ones, while a genuine
  "connected, no matches" result still returns empty columns without raising
  ([@m-samland](https://github.com/m-samland)).
- **`SphereDatabase.filter()`** – Composable, validated observation-table filtering, plus
  `view()` and a `columns` property. Per-column keyword criteria support equality,
  membership, comparison/`in`/`contains` tuples (e.g. `("<", 5)`), and a `public` keyword
  that restricts results to observations out of the ESO proprietary period. Missing
  values are excluded per criterion; `exclude_targets` drops named targets after the
  `target_list` restriction, and pre-computed boolean arrays remain the escape hatch for
  cross-column expressions ([@m-samland](https://github.com/m-samland)).
- **MOCAdb integration for stellar ages and young-association membership** – New
  `mocadb_matching` module cross-matches targets against MOCAdb (Gagné et al. 2026) via
  Gaia DR3 source IDs, adding 28 `MOCA_`-prefixed columns including association
  membership, BANYAN Σ probabilities, and adopted ages. The adopted association is the
  BANYAN Σ one where available and the literature one otherwise, with `MOCA_AID_SOURCE`
  recording which supplied each match. Optional dependency `pymysql`
  ([@m-samland](https://github.com/m-samland)) ([#107](https://github.com/m-samland/spherical/issues/107)).
- **`plot_trap_mosaics` console script** – Builds per-template detection-map and spectrum
  mosaics from a TRAP results tree, optionally annotated with SNR filtering and
  exposure/rotation metadata from the observation table. Supports
  `combined`/`detection`/`spectrum` content, single-file or `--batch-size`-split PDF/PNG
  output, and fixed-sigma or `--auto-scale` color limits. `--instrument` selects which
  observation table to read; it is inferred from an `IFS`/`IRDIS` segment in the results
  path when omitted ([@m-samland](https://github.com/m-samland)).
- **Pixi support** – `pixi.toml` with feature-based environments (`pipeline`, `notebook`,
  `docs`, `test`, `dev`, `dev-git`). `dev` installs `charis`/`trap` editable from local
  sibling checkouts for cross-package work; `dev-git` pulls both from git for working on
  `spherical` alone. The `pyproject.toml` pip workflow remains fully functional
  ([@m-samland](https://github.com/m-samland)) ([#102](https://github.com/m-samland/spherical/issues/102), [#112](https://github.com/m-samland/spherical/issues/112)).
- **Resume support for file-table generation** – `make_file_table()` writes new data to a
  `*_partial.csv` during processing and only updates the final output on success, so an
  interrupted run is continued simply by re-running it; already-fetched DP.IDs are
  skipped. New `resume` parameter, default `True`
  ([@m-samland](https://github.com/m-samland)) ([#105](https://github.com/m-samland/spherical/issues/105)).
- **51 Eridani astrometry regression tests** – `tests/regression/` (opt in with
  `-m regression`) pins the reported companion position of the 2015-09-24 IRDIS DB_K12 and
  IFS OBS_H datasets against frozen baselines in `tests/regression/data/`, and checks
  agreement with the published GRAVITY position.
  `tests/regression/data/51eri_astrometry_benchmark.md` documents the comparison
  ([@m-samland](https://github.com/m-samland)).

### 🔄 Changed
- **Table filenames name the mode instead of encoding a boolean** – The old
  `..._irdis_pol_{True,False}` scheme is replaced by the canonical modes `ifs`, `ifs_sam`,
  `irdis`, `irdis_polarimetry` and `irdis_sam`, derived by
  `database_utils.resolve_mode_name()` and matching the names used on Zenodo. **Existing
  local tables must be renamed or re-downloaded** ([@m-samland](https://github.com/m-samland)).
- **The pipeline environments cap `numpy<2.5`** – The bound comes from numba, pulled in
  transitively by trap; charis itself declares `numpy>=2,<3`. Without the cap the
  `linux-aarch64` solve fails. The database-only install is unaffected
  ([@m-samland](https://github.com/m-samland)).
- **The pipeline resumes by default; `force` replaces the `overwrite_*` flags** – Enabled
  steps whose outputs already exist on disk are skipped, so re-running over a growing
  target list is cheap and adding new targets just works. Recomputation is opt-in via the
  single `PipelineStepsConfig.force` field (`True`, or a set of step names that cascades
  to all downstream steps). A new `pipeline/step_registry.py` is the single source of
  truth for step log names and output-completion checks, shared by both instruments
  ([@m-samland](https://github.com/m-samland)) ([#121](https://github.com/m-samland/spherical/pull/121)).
- **Requires `trap >= 2.0.0`** – 2.0.0 is the first trap release carrying the astrometry
  behaviour and SPHERE anamorphism defaults that the frozen 51 Eri baselines were
  measured against, and it removed the deprecated `Reduction_parameters` path. Because
  `trap` is a git-URL dependency and cannot carry a PEP 508 floor, the minimum is
  enforced at import time with a clear error — which now also reminds you to reinstall an
  editable sibling checkout, whose recorded version is stamped at install time
  ([@m-samland](https://github.com/m-samland)).
- **Upstream charis changed the SPHERE hex bad-lenslet flagging** – The rule is now
  one-sided on inverse variance and masks about 40% fewer spaxels, and it is stable from
  frame to frame instead of flagging a different ~4.5% of the field each frame
  ([charis `fd35ece`](https://github.com/PrincetonUniversity/charis-dep/commit/fd35ece)).
  charis's hexagon-to-square resample also now propagates variance rather than applying
  the flux operator to the inverse variance
  ([charis #42](https://github.com/PrincetonUniversity/charis-dep/issues/42)), which
  corrects the absolute noise scale and makes flagged spaxels survive as exact zeros.
  **Existing reductions need a re-extraction to pick both up**
  ([@m-samland](https://github.com/m-samland)).
- **The monitoring scripts report instrument and pipeline separately** – `reduction_status`
  and `crash_reports` gained an `INSTR` column (IFS/IRDIS, derived from the observation
  band) and renamed the old `TYPE` column to `PIPELINE` with values `reduction`/`trap`,
  since IRDIS reductions were previously labelled `ifs`. `reduction_status --pipeline-type`
  is now `--pipeline {reduction,trap,all}`; the CSV columns follow the same names. Both
  scripts take `--instrument {ifs,irdis,all}`, and in `crash_reports` the exception tally
  follows the filter rather than the unfiltered set
  ([@m-samland](https://github.com/m-samland)).
- **Quieter TRAP batch runs** – `run_trap_on_observations` lowers the `trap` logger to
  `WARNING` (or `INFO` when `trap_config.processing.verbose`) before iterating and wraps
  the observation loop in a progress bar, so multi-target runs no longer flood the console
  ([@m-samland](https://github.com/m-samland)).
- **`SphereDatabase.observations_from_name_SIMBAD` accepts one or more names** – The
  method now takes a single name or a list/tuple, resolving each and returning the
  combined observations without duplicates; passing a list previously raised
  `AttributeError`. As part of the rewrite, `usable_only=True` now correctly returns only
  usable observations rather than all observations of any target with at least one usable
  observation ([@m-samland](https://github.com/m-samland)).
- **Gaia astrophysical parameters rounded and stored as `float32`** – The `GAIA_*` columns
  carried spurious precision from a float32→float64 promotion. Temperatures are now
  rounded to whole Kelvin and `logg`/`M_H`/`A_G` to two decimals, matching GSP-Phot's real
  precision, and all twelve columns are stored at the archive's native `float32`
  ([@m-samland](https://github.com/m-samland)).
- **Notebook dependencies switched to JupyterLab** – The `notebook` extra (and pixi
  feature) now installs `jupyterlab` instead of the classic `notebook` package. The unused
  `ipympl` and `ipydatagrid` were dropped; `ipywidgets` is kept because `tqdm.auto` uses it
  for widget progress bars ([@m-samland](https://github.com/m-samland)).
- **`examples/explore_database.ipynb` updated** – Follows the new mode naming, exposes an
  instrument/polarimetry/SAM selector so the SAM tables are reachable, and adds a section
  demonstrating the Gaia DR3 and MOCAdb enrichment columns
  ([@m-samland](https://github.com/m-samland)).
- **README rewritten** – Documents the IRDIS reduction as a first-class workflow alongside
  IFS, restructures installation into pip and pixi options, and consolidates the database
  table information into one section ([@m-samland](https://github.com/m-samland)).
- **Test suite reorganised by subject** – Tests now live in `tests/pipeline/`,
  `tests/database/` and `tests/regression/`, with cost expressed as markers (`remote_data`,
  `regression`, and a new `slow`) instead of a hand-maintained `--ignore` list. `addopts`
  deselects the expensive markers, so a bare `pytest` is offline and fast (~45s, previously
  19 minutes), and `testpaths` keeps collection out of `examples/`. New pixi tasks
  `test-pipeline`, `test-database`, `test-network`, `test-regression` and `test-all` select
  by path. The ESO resume tests replay from `pytest-recording` cassettes in
  `tests/database/cassettes/` instead of costing ~15 minutes against the live archive;
  `test_live_archive_still_responds` stays live behind `-m remote_data` as the API-drift
  canary ([@m-samland](https://github.com/m-samland)).

### 🗑️ Removed
- **`PipelineStepsConfig.overwrite_calibration` / `overwrite_bundle` /
  `overwrite_preprocessing` / `overwrite_trap`** – Superseded by `force`
  ([@m-samland](https://github.com/m-samland)) ([#121](https://github.com/m-samland/spherical/pull/121)).
- **The deprecated `FAILED_SEQ` / `_ready_flag` readiness path** – Superseded by
  `HCI_READY` ([@m-samland](https://github.com/m-samland)).
- **The custom `utils.progress` module** – Replaced by `tqdm.auto`, which detects
  notebook versus console more robustly and falls back correctly without `ipywidgets`
  ([@m-samland](https://github.com/m-samland)) ([#104](https://github.com/m-samland/spherical/issues/104)).
- **Callable masks in observation-table filtering** – `filter()` no longer accepts
  `lambda t: ...`; use a keyword criterion or a pre-computed boolean array
  ([@m-samland](https://github.com/m-samland)).
- **`pipeline.simplified_IRDIS_reduction`** – The single-script IRDIS prototype that
  preceded this release is superseded by the step-based `irdis_reduction` pipeline. It had
  no callers and no longer imported on scikit-image ≥ 0.19
  ([@m-samland](https://github.com/m-samland)).

### 🐛 Fixed
- **`make_file_table(cache=False)` did not actually disable caching** – the flag reached
  `Eso.get_headers` but not `Eso.query_instrument`, so a run explicitly asking for fresh
  data still resolved *which files exist* from astroquery's on-disk cache. Files added to a
  night that had already been queried were therefore invisible to an update, however the
  caller set `cache`. `query_eso_data` now passes it through — which does mean a full build
  issues more archive queries, since repeated identical queries within a run are no longer
  served from disk ([@m-samland](https://github.com/m-samland)).
- **TRAP's own progress never reached the target log** – nothing configured the `trap`
  logger tree, so its records died at `logging.lastResort` (level WARNING): a target that
  ran for hours left a `trap_reduction.log` holding only the pipeline's own bookend
  messages, and on a crash a traceback with no indication of what TRAP was doing.
  `processing.verbose=True` did not help — that only adds a stdout handler to the *pipeline*
  logger. New `bridge_library_logger()` routes the library's records into the same per-target
  files for the lifetime of that target, restoring the tree's level and handlers afterwards
  ([@m-samland](https://github.com/m-samland)).
- **A stale `trap_crash_report.txt` outlived the failure it described** – nothing removed it
  on a later successful run, so `crash_reports` kept flagging targets that had since been
  fixed. It is now cleared when the target starts ([@m-samland](https://github.com/m-samland)).
- **Observations sharing a target/band/night silently lost their log** –
  `get_pipeline_logger()` archived the log files before its "already configured" check and
  then returned a cached logger whose queue listener had already been stopped, so the
  second such observation wrote nothing and was left with no `reduction.jsonlog` — making
  it invisible to `aggregate_reduction_status`. The reuse check now runs first, and a
  logger whose listener is gone is rebuilt ([@m-samland](https://github.com/m-samland)).
- **A single failed target could abort a whole TRAP batch** – the prologue of
  `run_trap_on_observation()` (instrument lookup, path construction, `validate_force`,
  logger setup) runs before its own try/except. `run_trap_on_observations()` now isolates
  each observation and continues with the next
  ([@m-samland](https://github.com/m-samland)).
- **TRAP post-processing crashed on trap's now-immutable reduction config** – `run_trap.py`
  built a legacy `Reduction_parameters` via the deprecated
  `TrapConfig.get_reduction_parameters()` and then mutated `result_folder` in place, which
  raised `FrozenInstanceError` once `trap` froze the dataclass and emitted three
  `DeprecationWarning`s per reduction. Both the reduction and detection paths now use
  `trap_config.reduction.merge(result_folder=…)`. `examples/ifs_reduction_template.py` was
  migrated to the same `.merge()` pattern
  ([@m-samland](https://github.com/m-samland)) ([#115](https://github.com/m-samland/spherical/issues/115)).
- **`crash_reports` reported `unknown` for every TRAP crash** – The dataset was parsed only
  from the reduction crash report's wording, so TRAP reports — which open with a different
  sentence — lost their `target/band/night` identifier. Both aggregators also sized their
  table columns to a fixed padding, which long target names and six-character bands pushed
  out of alignment with the header; widths now follow the content
  ([@m-samland](https://github.com/m-samland)).
- **Raw-file resolution and `.fits.Z` remnants** – An interrupted `retrieve_data(unzip=True)`
  left `.fits.Z` files that the `SPHER.*.fits` glob missed, so those DP.IDs were marked
  missing and the preprocess loader failed with a cryptic `Empty filename: ''`. The glob now
  covers both extensions, leftover `.fits.Z` files are decompressed at the end of the
  download step (astropy reads them ~3000× slower), and `update_observation_file_paths`
  raises `FileNotFoundError` at the boundary with a message that says whether to retry the
  download or enable it ([@m-samland](https://github.com/m-samland)).
- **`IndexError` in `filter_for_science_frames()` for IRDIS non-polarimetry and SAM modes** –
  The `DPR_TECH` match array was computed once before the polarimetry filter shrank the
  table, so the later SAM filter applied a stale full-length mask. It is now recomputed
  before each filtering step ([@m-samland](https://github.com/m-samland)).
- **ESO header retrieval hung indefinitely on a flaky link** – `make_file_table()` issued
  archive requests with no timeout. The session now sets a (30 s connect, 120 s read)
  timeout and mounts urllib3 retries with backoff, so transient disconnects self-heal
  within the run ([@m-samland](https://github.com/m-samland)).
- **Package data files were missing from built wheels** – `simbad_tap_query.adql`, the
  IRDIS/CPI filter curves, and `ifu_mask.fits` were loaded at runtime but absent from
  `package-data`, so non-editable installs raised `FileNotFoundError`
  ([@m-samland](https://github.com/m-samland)).
- **`ROTATION` stores `np.nan` instead of a `-10000` sentinel** – Non-applicable or failed
  derotation now round-trips to a masked column and is treated as missing by filtering.
  Takes effect on the next database rebuild ([@m-samland](https://github.com/m-samland)).
- **Cosmetic warnings silenced** – The astropy `VerifyWarning` and `MergeConflictWarning`
  that flooded long file-table updates and SIMBAD target-table builds (each batched query
  carries its own result identifier in `.meta`, so every `vstack` warned), and the
  `All-NaN slice encountered` `RuntimeWarning` from PSF centering on IRDIS DBI, where dead
  detector regions overlap in both channels. All were harmless — the PSF path calls
  `np.nan_to_num` on the very next line ([@m-samland](https://github.com/m-samland)).
- **Spurious "No usable science frames" warnings during observation-table generation** –
  `SKY` frames were in the science-frame matching pool but never handled by
  `select_primary_science_frames()`, creating ghost observation groups that always failed
  ([@m-samland](https://github.com/m-samland)).
- **`run_cube_header_update` crashed outside a git repository** – Git metadata collection
  now targets the spherical source tree directly and falls back to `"unknown"` when git is
  unavailable, so the pipeline runs from network filesystems and HPC scratch
  ([@m-samland](https://github.com/m-samland)) ([#101](https://github.com/m-samland/spherical/issues/101)).
- **IFS step ordering crashed at the cube header update** – `run_frame_info_computation` now
  runs before `run_cube_header_update`, resolving `FileNotFoundError` for the
  `frames_info_*.csv` files. Also adds defensive handling for missing frame-info CSVs and
  glob escaping for target names with special characters
  ([@m-samland](https://github.com/m-samland)) ([#97](https://github.com/m-samland/spherical/issues/97)).
- **CI collected no tests at all** – five test modules imported `scipy`/`photutils` without a
  guard, so the `pip install ".[test]"` environment aborted collection with exit code 2
  before running anything; the database coverage CI was supposed to provide had been
  silently absent. They now carry a module-level `pytest.importorskip`, and
  `test_connection_failure_raises` (which patches `pymysql`, shipped in the `mocadb` extra
  rather than `test`) skips cleanly instead of failing
  ([@m-samland](https://github.com/m-samland)).

---

## [2.1.3] - 2026-02-19

### ✨ Added
- **Intermediate file cleanup** – Added functionality to clean up intermediate pipeline files after successful reduction. The clean-up can be triggered using the methods outlined in the example reduction script ([@m-samland](https://github.com/m-samland)).

### Fixed
- **Fixed ESO data download crash when keyring is unavailable** – Fixed crash when downloading proprietary ESO data on machines without an installed keyring. The `store_password` default is now `False`, and keyring interactions are wrapped in try/except for robustness ([@m-samland](https://github.com/m-samland)) ([#94](https://github.com/m-samland/spherical/issues/94)).
- **Fixed SIMBAD target name resolving** – Fixed SIMBAD query failures caused by using the outdated column name `MAIN_ID` instead of `main_id`, which could cause reductions to be skipped for certain targets ([@m-samland](https://github.com/m-samland)) ([#95](https://github.com/m-samland/spherical/issues/95)).

---

## [2.1.2] - 2025-08-10

### ✨ Added
- **Comprehensive TRAP pipeline logging** – Added structured logging to TRAP (Temporal Reference Analysis of Planets) pipeline functions (`run_trap_on_observation` and `run_trap_on_observations`) following the same schema as IFS reduction steps. Includes session tracking, error handling with crash reports (`trap_crash_report.txt`), and debug logging for troubleshooting. Enhanced aggregation scripts (`aggregate_crash_reports.py`, `aggregate_reduction_status.py`) to support unified monitoring of both IFS and TRAP logs with pipeline type detection and flexible filtering options ([@m-samland](https://github.com/m-samland)) ([#91](https://github.com/m-samland/spherical/issues/91)).

### Changed
- **Major README.md overhaul** – Major overhaul and expansion of `README.md`: clarifies the usage model, installation (including environment setup and dependencies), quick start, helper scripts, testing, and contribution guidelines. Adds explicit notes about the database containing metadata only, and emphasizes the script-driven nature of the IFS pipeline ([@m-samland](https://github.com/m-samland)).

### Fixed
- **Fixed binary star naming resolution** – Improved target lookup to automatically try both naming variations (with and without "A" suffix) during local database search. Now searches for example for both `"HD 95086"` and `"HD 95086 A"` automatically before falling back to SIMBAD queries, resolving lookup failures for binary star systems regardless of naming convention used ([@m-samland](https://github.com/m-samland)) ([#90](https://github.com/m-samland/spherical/issues/90)).

---

## [2.1.1] - 2025-07-23

### ✨ Added
- **Added unit tests for configuration** – Added comprehensive test suite (`test_config.py`) to verify pipeline configuration functionality. The configuration tests validate dataclass instantiation, default values, merge operations, resource management, and directory path handling across all pipeline configuration classes ([@m-samland](https://github.com/m-samland)). 

### Fixed
- **Include missing pipeline module** – Fixed packaging issue by adding the missing `pipeline/steps/cube_header_update.py` module to the repository ([@m-samland](https://github.com/m-samland)).
- **Fixed astropy compound model parameter extraction** – Fixed AttributeError when `fit_background=True` by properly accessing parameters with `_0`/`_1` suffixes in compound models ([@m-samland](https://github.com/m-samland)) ([#86](https://github.com/m-samland/spherical/issues/86)).
- **Fixed single flux frame edge case handling** – Fixed KeyError when processing observations with only one flux frame by ensuring proper column structure in flux calibration indices and proper initialization of discontinuity arrays for edge cases ([@m-samland](https://github.com/m-samland)) ([#87](https://github.com/m-samland/spherical/issues/87)).
- **Fixed array dimensional consistency for single center observations** – Resolved IndexError and broadcasting issues when processing observations with only one temporal frame by implementing conditional squeezing in `extract_satellite_spot_stamps()` to return appropriate dimensions for flux PSF (4D) vs satellite spots (5D) use cases ([@m-samland](https://github.com/m-samland)) ([#88](https://github.com/m-samland/spherical/issues/88)).

---

## [2.1.0] - 2025-07-21

### ✨ Added
- **Typed dataclass configuration system** – Refactored the IFS reduction pipeline configuration from plain dictionaries to a comprehensive typed dataclass-based architecture. The new system provides type safety, IDE autocompletion, intelligent defaults, and better maintainability while maintaining backward compatibility through `as_plain_dicts()` method ([@m-samland](https://github.com/m-samland)).
- **Centralized logging infrastructure** – Implemented uniform logging schema across all IFS pipeline steps with `@optional_logger` decorator, structured logging with automatic context injection, and multiprocessing-safe `QueueHandler` mechanism. All pipeline steps now support consistent, testable, and aggregatable logging ([@m-samland](https://github.com/m-samland)).
- **Pipeline monitoring scripts** – Added installable command-line tools `crash_reports` and `reduction_status` to aggregate and summarize pipeline execution across large reduction campaigns. Scripts parse structured logs and crash reports to provide dataset completion status, exception frequency analysis, and CSV export capabilities ([@m-samland](https://github.com/m-samland)).
- **Multi-observation wrapper function** – Added `execute_targets()` wrapper function to `ifs_reduction.py` that processes multiple SPHERE observations sequentially using the same configuration, simplifying batch processing workflows and removing explicit loops from user scripts ([@m-samland](https://github.com/m-samland)).
- **TRAP post-processing wrapper functions** – Added `run_trap_on_observation()` and `run_trap_on_observations()` wrapper functions to `pipeline/run_trap.py` module, providing a consistent modular interface for TRAP post-processing that mirrors the IFS reduction pipeline structure. Refactored inline TRAP processing code into reusable functions, improving maintainability and code organization ([@m-samland](https://github.com/m-samland)).
- **Comprehensive documentation** – Added NumPy-style docstrings throughout the `ifs_reduction.py` module following astronomical software documentation standards, including scientific context, parameter units, wavelength specifications, and coordinate system references ([@m-samland](https://github.com/m-samland)).
- **Added stellar cluster age matching** – Added a convenient way to match an observation list to stellar cluster ages from [Hunt+24](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A..42H/abstract). This allows, in a limited way, to add age data to the data selection criteria. This TAP ADQL infrastructure can be used for other Vizier catalogs in the future for more complete stellar age coverage ([@m-samland](https://github.com/m-samland)).
- **Update FITS header for IFS outputs** – Populate the extracted IFS data with meta data about pipeline versions and other useful information to ensure archival usefulness. Implemented as a dedicated `cube_header_update` pipeline step that runs automatically after `bundle_output` and updates FITS headers with comprehensive metadata including software versions, processing parameters, git repository information, and provenance data. The step can be configured via the pipeline configuration system ([@lwelzel](https://github.com/lwelzel), [@m-samland](https://github.com/m-samland)) ([#66](https://github.com/m-samland/spherical/pull/66)).
- **Pipeline cleanup utilities** – Added dedicated cleanup module (`pipeline/cleanup.py`) with utilities to check pipeline completion status and manage storage by cleaning intermediate files. Includes `check_cube_building_success()` to verify CHARIS pipeline and bundling completion, and cleanup functions for raw data (`clean_raw_data()`), extracted cubes (`clean_extracted_cubes()`), wavelength calibrations (`clean_wavelength_calibrations()`), and wrapper function (`clean_all_intermediate_files()`) for batch operations. All functions support dry-run mode for safety and provide detailed size reporting ([@m-samland](https://github.com/m-samland)).

### Changed
- **Pipeline configuration architecture** – Introduced `CalibrationConfig`, `ExtractionConfig`, `PreprocConfig`, `DirectoryConfig`, and `Resources` dataclasses with merge functionality, centralized CPU allocation, and automatic path resolution. Factory method `defaultIFSReduction()` provides easy default configuration creation ([@m-samland](https://github.com/m-samland)).
- **Pipeline logging standardization** – Refactored all pipeline steps to use centralized logging with automatic injection of static context fields (`target`, `band`, `night`), structured status logging (`status`: `"success"`/`"failed"`), and eliminated `print()` statements in favor of proper log levels. Enhanced multiprocessing safety and debugging capabilities ([@m-samland](https://github.com/m-samland)).
- **Database class naming** – Updated `Sphere_database` to `SphereDatabase` following PEP8 naming conventions for improved code consistency ([@m-samland](https://github.com/m-samland)).
- **Modular pipeline architecture** – Completely restructured the IFS data reduction pipeline into discrete, self-contained modules located in `pipeline/steps/`. Each processing step (wavelength calibration, cube extraction, astrometric calibration, etc.) is now an independent module with a single function call containing all required logic. This modularization significantly improves code maintainability, enables comprehensive unit testing, and provides a future-proof architecture for pipeline extensions ([@m-samland](https://github.com/m-samland)).
- **Depdenencies update** – Incremented astropy version to >=7.1.
- **Output folder decluttered** – Additional outputs generated by the IFS pipeline, which may not be needed by the average user and most post-processing pipelines have been moved to an `additional_output` directory.

### Fixed
- **Updated flux PSF** – Updated the file used by TRAP as PSF model for the companion. Previously, an a file that did not drop the first (bad) frame was used. All PSF frames are normalized to the same mean value before being combined now.
- **Improved flux PSF finding** – Improved the way that the flux PSF is detected in the IFS cubes. Sometimes the detection failed, especially for the first bad frame, which broke the pipeline for some targets.

## [2.0.0] – IRDIS Support and Pipeline Enhancements (2025-05-18)

### ✨ Added
- **IRDIS dual-band imaging data support** – The observation database now includes VLT/SPHERE IRDIS dual-band imaging (DBI) sequences for seamless querying and retrieval via the pipeline ([@m-samland](https://github.com/m-samland)) ([#53](https://github.com/m-samland/spherical/issues/53)).
- **IRDIS polarimetry data support** – Added SPHERE/IRDIS dual-polarization imaging (DPI) observations to the database, enabling search and download of IRDIS polarimetric data through the same interface ([@m-samland](https://github.com/m-samland)) ([#49](https://github.com/m-samland/spherical/issues/49)).
- **Unified data download module** – The data download process was refactored into an independent module that handles both **IFS** and **IRDIS** instrument modes consistently. It now supports proper folder management, avoids redundant downloads, and permits custom keyword filtering for additional file types ([@m-samland](https://github.com/m-samland)) ([#60](https://github.com/m-samland/spherical/issues/60)).

### Changed
- **Refactored data download architecture** – The download functionality was overhauled to robustly handle multiple instrument modes (IFS and IRDIS) and improve reliability for large batch fetches ([@m-samland](https://github.com/m-samland)) ([#60](https://github.com/m-samland/spherical/issues/60)).
- **New sequence readiness flag** – Replaced the `FAILED_SEQ` status flag with a clearer `HCI_READY` flag to mark sequences that are ready for high-contrast imaging analysis in the database ([@m-samland](https://github.com/m-samland)) ([#56](https://github.com/m-samland/spherical/issues/56)).
- **Optimized SIMBAD cross-matching** – Pre-selects candidate targets based on sky position (Healpix) before querying SIMBAD, reducing false matches and lowering query overhead ([@m-samland](https://github.com/m-samland)) ([#54](https://github.com/m-samland/spherical/issues/54)).
- **Incremental header saving** – During database construction, header tables are now saved incrementally to avoid memory issues and prevent data loss if a process is interrupted ([@m-samland](https://github.com/m-samland)) ([#52](https://github.com/m-samland/spherical/issues/52)).
- **Updated external pipeline dependency** – Pointed the SPHERE/CHARIS IFS spectral extraction pipeline to its latest **main** branch for long-term stability improvement ([@m-samland](https://github.com/m-samland)) ([#47](https://github.com/m-samland/spherical/issues/47)).
- **Module renaming for clarity** – Renamed the `sphere_database` module to `database` to simplify the package structure and usage ([@m-samland](https://github.com/m-samland)) ([#44](https://github.com/m-samland/spherical/issues/44)).
- **Metadata computation refactor** – Moved calculation of observational metadata (e.g. exposure times, parallactic rotation) into the database module for a cleaner pipeline workflow ([@m-samland](https://github.com/m-samland)) ([#42](https://github.com/m-samland/spherical/issues/42)).

### Fixed
- **Batch table overwrite bug** – Fixed an issue where creating the file table in batches could overwrite results from earlier batches ([@m-samland](https://github.com/m-samland)) ([#50](https://github.com/m-samland/spherical/issues/50)).
- **Parallactic angle calculation (important)** – Corrected the `ROTATION` (parallactic angle change) values in observation summary tables to reflect true field rotation ([@m-samland](https://github.com/m-samland)) ([#46](https://github.com/m-samland/spherical/issues/46)).
- **PSF extraction in unocculted frames** – Resolved a bug causing the PSF-finding routine to fail on non-coronagraphic images when the telescope pointing offset changed between frames ([@m-samland](https://github.com/m-samland)) ([#45](https://github.com/m-samland/spherical/issues/45)).
- **Spectral cube assembly** – The cube building step now handles incorrect NDIT header values gracefully, instead of aborting the process ([@m-samland](https://github.com/m-samland)) ([#43](https://github.com/m-samland/spherical/issues/43)).
- **Run extract_cube without previous steps** – Fixed a file path renaming issue that would prevent the `extract_cube`-step to run, when not running the previous steps in the same session ([@m-samland](https://github.com/m-samland)) ([#41](https://github.com/m-samland/spherical/issues/41)).
- **Timing table overwrite** – Ensured that the `compute_times` function no longer overwrites its input timing table, preserving original data ([@m-samland](https://github.com/m-samland)) ([#40](https://github.com/m-samland/spherical/issues/40)).


## [1.1.1] - 2025-04-10

### ✨ Added
- Documented that Python 3.13 is not yet supported when installing the pipeline dependencies (in the README) ([@m-samland](https://github.com/m-samland)) ([#32](https://github.com/m-samland/spherical/pull/32)).
- Added multi-processing to find the star center using satellite spots for each frame ([@m-samland](https://github.com/m-samland)) ([#31](https://github.com/m-samland/spherical/pull/31)).

### Changed
- Improved the plot of star center evolution over time and wavelength; this is now an independent step ([@m-samland](https://github.com/m-samland)) ([#38](https://github.com/m-samland/spherical/pull/38)).
- Moved `find_star` to its own module ([@m-samland](https://github.com/m-samland)) ([#37](https://github.com/m-samland/spherical/pull/37)).
- Refactored `compute_times` in the pipeline to include `DIT_DELAY` ([@m-samland](https://github.com/m-samland)) ([#36](https://github.com/m-samland/spherical/pull/36)).
- Renamed `calibrate_center` to `find_star` ([@m-samland](https://github.com/m-samland)) ([#22](https://github.com/m-samland/spherical/pull/22)).

### Fixed
- Fixed multiple bugs to enable running each step of the IFS reduction template ([@m-samland](https://github.com/m-samland)) ([#27](https://github.com/m-samland/spherical/pull/27), [#29](https://github.com/m-samland/spherical/pull/29)).

## [1.1.0] - 2025-04-04

### ✨ Added
- Added documentation to the high-level database functions describing the content of all tables ([@m-samland](https://github.com/m-samland)) ([#22](https://github.com/m-samland/spherical/pull/22)).
- Added file size estimates for each SPHERE FITS file to support download size estimates per observation sequence ([@m-samland](https://github.com/m-samland)) ([#16](https://github.com/m-samland/spherical/pull/16)).
- Added progress bar for database table creation ([@lwelzel](https://github.com/lwelzel)) ([#2](https://github.com/m-samland/spherical/pull/2)).
- Added support for batched queries ([@lwelzel](https://github.com/lwelzel)) ([#2](https://github.com/m-samland/spherical/pull/2)).
- Added changelog ([@m-samland](https://github.com/m-samland)) ([#6](https://github.com/m-samland/spherical/pull/6)).
- Added CI and `ruff` linting workflow ([@m-samland](https://github.com/m-samland)) ([#5](https://github.com/m-samland/spherical/pull/5)).
- Added basic end-to-end tests for the database ([@m-samland](https://github.com/m-samland)) ([#4](https://github.com/m-samland/spherical/pull/4)).
- Added issue and PR templates ([@m-samland](https://github.com/m-samland)) ([#3](https://github.com/m-samland/spherical/pull/3)).

### Changed
- Renamed "master file table" to simply "file table" ([@m-samland](https://github.com/m-samland)) ([#22](https://github.com/m-samland/spherical/pull/22)).
- Filtered SIMBAD stars associated with observation sequences to require proper motion. Excluded stars with parallaxes implying distances greater than 1 kpc (unlikely to be direct imaging targets) ([@m-samland](https://github.com/m-samland)) ([#13](https://github.com/m-samland/spherical/pull/13)).
- Switched to using an `astropy` function for proper motion propagation ([@m-samland](https://github.com/m-samland)) ([#12](https://github.com/m-samland/spherical/pull/12)).
- Separated pipeline dependencies from the database. To analyze data, use:  
  ```bash
  pip install ".[pipeline]"
  ```  
  ([@m-samland](https://github.com/m-samland)) ([#9](https://github.com/m-samland/spherical/pull/9)).
- Added summary information (e.g., total exposure time, parallactic angle change) to the observation sequence table with values rounded to sensible precision ([@m-samland](https://github.com/m-samland)) ([#15](https://github.com/m-samland/spherical/pull/15)).

### Fixed
- Fixed compatibility with newer `astroquery` versions ([@lwelzel](https://github.com/lwelzel), [@m-samland](https://github.com/m-samland)) ([#2](https://github.com/m-samland/spherical/pull/2), [#14](https://github.com/m-samland/spherical/pull/14)).
- Fixed an issue where the number of integrations (`NDIT`) was read under two different names, causing downstream errors ([@m-samland](https://github.com/m-samland)) ([#10](https://github.com/m-samland/spherical/pull/10)).

---

## [1.0.0] - 2024-12-27

### Added
- Initial release.

### Changed
- Initial implementation of core functionality.

### Fixed
- No known issues.

[Unreleased]: https://github.com/m-samland/spherical/compare/v3.0.0...HEAD  
[3.0.0]: https://github.com/m-samland/spherical/compare/v2.1.3...v3.0.0  
[2.1.3]: https://github.com/m-samland/spherical/compare/v2.1.2...v2.1.3  
[2.1.2]: https://github.com/m-samland/spherical/compare/v2.1.1...v2.1.2  
[2.1.1]: https://github.com/m-samland/spherical/compare/v2.1.0...v2.1.1  
[2.1.0]: https://github.com/m-samland/spherical/compare/v2.0.0...v2.1.0  
[2.0.0]: https://github.com/m-samland/spherical/compare/v1.1.1...v2.0.0  
[1.1.1]: https://github.com/m-samland/spherical/compare/v1.1.0...v1.1.1  
[1.1.0]: https://github.com/m-samland/spherical/compare/v1.0.0...v1.1.0  
[1.0.0]: https://github.com/m-samland/spherical/releases/tag/v1.0.0

"""IRDIS reduction driver — Phases 1 + 3 + 4.

Runs the IRDIS pipeline up to and including preprocessing. What each phase
produces:

- **Phase 1 — download** (``steps.download_data``): pulls raw ``CORO``,
  ``CENTER``, ``FLUX``, ``FLAT``, and ``BG_SCIENCE`` FITS files for each
  observation from the ESO archive into ``raw_directory``. Idempotent — files
  already present locally are skipped, so re-runs are cheap.
- **Phase 3 — master calibration** (``steps.irdis_calibration``): from the
  ``FLAT`` and ``BG_SCIENCE`` frames, builds ``master_flat.fits``,
  ``master_background.fits``, and ``badpixel_map.fits`` under
  ``{reduction_directory}/IRDIS/calibration/{filter}/{date}/``. Internal-guard
  step — skips itself when all three outputs already exist.
- **Phase 4 — science preprocess** (``steps.preprocess_irdis``): per
  observation, per frame type (CORO/CENTER/FLUX), scaled-background
  subtraction + flat division + analytic ivar + NaN-safe bad-pixel
  replacement. Optional per-frame transient sigma-clip on non-FLUX (disabled
  by default; see the ``irdis_preprocessing`` block below). Writes 8 files
  under ``{reduction_directory}/IRDIS/observation/{target}/{filter}/{date}/converted/``:
  ``{coro,center,flux}_cube.fits``, matching ``*_ivar_cube.fits``,
  ``wavelengths.fits``, and ``badpixel_map.fits``. Gated by ``should_run`` —
  skips when all 8 outputs already exist. Cube axis order matches the IFS
  pipeline: ``(n_wave=2, n_time, ny, nx)``.

Phases 5 (shared-step generalization: find_centers, process_extracted_centers,
spot_to_flux) and 6 (TRAP integration) are not wired yet and remain disabled
below.

Inspecting results after a run:

    ls -1 ~/data/sphere/reduction/IRDIS/calibration/*/*/                     # Phase 3
    ls -1 ~/data/sphere/reduction/IRDIS/observation/*/*/*/converted/         # Phase 4

Quick FITS inspection:

    python -c "
    from astropy.io import fits
    import numpy as np
    from pathlib import Path
    for p in sorted(Path.home().glob('data/sphere/reduction/IRDIS/observation/*/*/*/converted/*_cube.fits')):
        data = fits.getdata(p)
        print(f'{p.name}: shape={data.shape}, dtype={data.dtype}, med={np.nanmedian(data):.3g}')
    "

For a persistent validation report with per-channel PNG previews, see the
plan's Task 7 Step 4 snippet in
``docs/superpowers/plans/2026-07-18-irdis-reduction-phase4-preprocess-step.md``.

Force recompute (e.g., after editing a step) via
``config.steps.force = {"preprocess_irdis"}`` — force cascades to every
downstream step in ``IRDIS_STEP_ORDER``.
"""
from pathlib import Path

from astropy.table import Table

from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IRDISReductionConfig

TARGET_LIST = ["* bet Pic"]


def main():
    # =================== CONFIGURATION ===================
    config = IRDISReductionConfig()

    config.set_ncpu(4)

    # Turn every step off, then explicitly enable the three phases that are wired.
    # `disable_all_ifs_steps` covers all shared steps (compute_frames_info,
    # find_centers, etc.); `disable_all_irdis_steps` covers those plus
    # `irdis_calibration` and `preprocess_irdis`, which default to True.
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()

    config.steps = config.steps.merge(
        download_data=True,        # Phase 1 — idempotent (skips if raw files present)
        irdis_calibration=True,    # Phase 3 — internal-guard (skips if outputs present)
        preprocess_irdis=True,     # Phase 4 — should_run-gated on the 8 converted/ outputs
        run_trap_reduction=False,
        run_trap_detection=False,
    )

    # ===== ESO archive credentials =====
    config.preprocessing = config.preprocessing.merge(
        eso_username=None,
        store_password=False,
        delete_password_after_reduction=False,
    )

    # ===== IRDIS-specific preprocessing knobs (Phase 4) =====
    # All fields have production-tuned defaults. Uncomment / edit only what
    # you need. Every setting merges into `config.irdis_preprocessing` via the
    # frozen-dataclass `.merge(...)` pattern.
    config.irdis_preprocessing = config.irdis_preprocessing.merge(
        # --- Bad-pixel replacement (vectorized K-nearest fill) ---
        # fix_badpix=True,           # Interpolate bpm pixels + set ivar=0 there.
        #                             # Turn off only for algorithmic ablation.

        # --- Per-frame transient sigma-clip ---
        # transient_nsigma=0.0,      # DEFAULT DISABLED. See the config comment
        #                             # in pipeline_config.py for the rationale
        #                             # (dominated by AO speckle chatter on real
        #                             # data, ~25% of wall time). Set to 8.0 if
        #                             # cube medians show visible CR streaks.
        #                             # Non-FLUX only regardless.

        # --- Star / PSF exclusion mask for scaled-background fit ---
        # star_mask_radius=285,      # px. K-band beta-Pic-calibrated; covers
        #                             # the AO-corrected halo out to where the
        #                             # image is bg-dominated.
        # flux_star_mask_radius=150, # px. Smaller because the FLUX PSF is
        #                             # compact off the coronagraph.

        # --- Anamorphism (default OFF; factor always recorded in the header) ---
        # correct_anamorphism=False, # True → scale y-axis by anamorphism_factor
        #                             # via cubic spline (scipy.ndimage.zoom).
        # anamorphism_factor=1.0062, # SPHERE-measured; do not tune per-target.

        # --- Optional crop around the star (default OFF; validate on full frames first) ---
        # crop=False,
        # crop_size=512,             # side of the square crop, px.
        # crop_center=None,          # (x, y) per-half; None → nominal position.

        # --- Detector constants (DO NOT TOUCH unless SPHERE hardware changes) ---
        # gain=1.75,                 # e-/ADU
        # read_noise=4.4,            # e-
    )

    # ===== Directory layout =====
    config.directories.base_path = Path.home() / "data/sphere"
    config.directories.raw_directory = config.directories.base_path / "data"
    config.directories.reduction_directory = config.directories.base_path / "reduction"

    database_directory = Path.home() / "data/sphere/database"

    instrument = "irdis"

    table_of_observations = Table.read(
        database_directory / f"table_of_observations_{instrument}.fits"
    )
    table_of_files = Table.read(
        database_directory / f"table_of_files_{instrument}.csv"
    )

    # ---------------------Database setup-----------------------------------------#
    database = SphereDatabase(
        table_of_observations, table_of_files, instrument=instrument
    )

    observation_table = database.filter(
        target_list=TARGET_LIST,
        TOTAL_EXPTIME_SCI=(">", 30),
        DEROTATOR_MODE="PUPIL",
        HCI_READY=True,
    )[:1]  # Limit to one observation for testing
    print(observation_table)

    observations = database.retrieve_observation_metadata(observation_table)

    execute_targets(observations=observations, config=config)


# IMPORTANT: keep every side-effecting call above (Table.read, database.filter,
# retrieve_observation_metadata, print, execute_targets) inside `main()` and
# guard the entry point with `if __name__ == "__main__":`. Phase 4 uses
# `ProcessPoolExecutor` with the `spawn` start method (macOS/Windows default),
# which re-imports this file into every worker process. Anything at module
# scope would run once PER worker at pool startup — meaning re-reads of the
# observation table, re-prints of the filtered rows, and one-shot tqdm bars
# from `retrieve_observation_metadata` firing 4×, 8×, ... times before the
# real preprocess bar even starts. Keeping setup inside `main()` gives workers
# a clean import.
if __name__ == "__main__":
    main()

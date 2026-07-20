"""IRDIS reduction driver — end-to-end template.

Runs download → master calibration → preprocess → shared downstream steps
(centering, flux PSF, spot-to-flux) → TRAP reduction + detection for the
observations selected below. Outputs land under
``{reduction_directory}/IRDIS/{calibration,observation,trap}/…``.

Force a rerun with ``config.steps.force = {"<step_name>"}`` — force cascades
to every downstream step in ``IRDIS_STEP_ORDER``.
"""
from pathlib import Path

from astropy.table import Table
from trap.parameters import StellarParameters, trap_config_for_irdis

from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IRDISReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations

TARGET_LIST = ["51 Eridani"]


def main():
    # =================== CONFIGURATION ===================
    config = IRDISReductionConfig()

    # ===== CPU budget =====
    # `set_ncpu(N)` sets a master budget that populates the per-step CPU counts
    # (extract, center, calibration, trap). Override individual step counts on
    # `config.resources` after this call if you want an asymmetric split.
    config.set_ncpu(4)
    # config.resources.ncpu_trap    = 8     # TRAP is IO/memory-heavy; give it more
    # config.resources.ncpu_calib   = 2
    # config.resources.ncpu_extract = 4
    # config.resources.ncpu_center  = 2

    # Turn every step off, then explicitly enable the three phases that are wired.
    # `disable_all_ifs_steps` covers all shared steps (compute_frames_info,
    # find_centers, etc.); `disable_all_irdis_steps` covers those plus
    # `irdis_calibration` and `preprocess_irdis`, which default to True.
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()

    config.steps = config.steps.merge(
        # Phase 1 / 3 / 4 — reduction path
        download_data=True,        # idempotent (skips if raw files present)
        irdis_calibration=True,    # internal-guard (skips if outputs present)
        preprocess_irdis=True,     # should_run-gated on the 8 converted/ outputs
        # Phase 5 — shared downstream steps (instrument-dispatched)
        compute_frames_info=True,
        cube_header_update=True,
        find_centers=True,
        process_extracted_centers=True,
        plot_image_center_evolution=True,
        calibrate_spot_photometry=True,
        calibrate_flux_psf=True,
        spot_to_flux=True,
        # Phase 6 — TRAP post-processing
        run_trap_reduction=True,
        run_trap_detection=True,
    )

    # ===== ESO archive credentials =====
    config.preprocessing = config.preprocessing.merge(
        eso_username=None,
        store_password=False,
        delete_password_after_reduction=False,
    )

    # ===== Master-calibration knobs (Phase 3) =====
    # Defaults are production-tuned on the DBI reference set. Only touch when
    # the calibration frames are unusually noisy or the flat has an atypical
    # response range.
    config.calibration = config.calibration.merge(
        # combination_method="median",         # 'median' or 'mean'
        # flat_badpix_sigma=5.0,               # σ threshold on flat outliers
        # background_badpix_sigma=5.0,         # σ threshold on background outliers
        # flat_relative_response_min=0.5,      # flag pixels with <50% response
        # flat_relative_response_max=1.5,      # flag pixels with >150% response
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

    # ===== Optional TRAP inputs =====
    # Enable to hand extra per-frame information to trap. Each flag is safe to
    # leave on for observations that don't produce the corresponding file
    # (INFO log + no-op fallback), but explicit is better for reproducibility.
    config.pass_inverse_variance_to_trap = True                # ivar cube (default True); improves noise weighting
    config.pass_center_outliers_as_bad_frames_to_trap = True   # union of per-channel waffle-fit outliers → trap bad_frames (waffle only)
    config.pass_amplitude_modulation_to_trap = True            # continuous-waffle only; loads spot_amplitude_variation.fits

    # ===== Per-target stellar parameters =====
    # Default True → pull effective temperature (and Gaia logg/[Fe/H]) per
    # observation from the observation-table's Gaia enrichment columns and
    # fall back to a spectral-type→Teff Mamajek table, then to trap_config's
    # StellarParameters. Set False to force every target to use the same
    # trap_config values.
    # config.use_gaia_stellar_parameters = True

    # ===== Coronagraph transmission =====
    # Default True → inject the packaged N_ALC_JYH_S transmission curve
    # (H23 file works for K12 too — the coronagraph is achromatic in the DBI
    # working range) as trap's `coronagraph_transmission`, correcting the
    # forward-model attenuation near the coronagraph. Setting False leaves
    # small-separation contrasts under-estimated.
    # config.apply_coronagraph_transmission = True

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
        # NIGHT_START=("2017-09-27"),
    )  # Limit to one observation for testing
    print(observation_table)

    observations = database.retrieve_observation_metadata(observation_table)

    execute_targets(observations=observations, config=config)

    # ===== Phase 6 — TRAP post-processing =====
    trap_config = trap_config_for_irdis()

    # --- TRAP reduction knobs ---
    # The IRDIS-flavored defaults set search_region_outer_bound=200 (~K-band AO
    # cutoff at 12.25 mas/px), inner_bound=1 (coronagraph IWA is enforced by
    # the injected coronagraph_transmission curve), temporal_model=True,
    # spatial_model=False, right_handed=False. Override here per-observation.
    trap_config.reduction = trap_config.reduction.merge(
        # search_region_inner_bound=1,
        # search_region_outer_bound=200,
        # yx_known_companion_position=None,     # e.g. [-35.95, -8.43] for 51 Eri b
        # temporal_model=True,
        # spatial_model=False,
        # right_handed=False,                   # False for SPHERE DC angles
        # estimate_noise_from_data=False,       # ignored when ivar cube is passed
        # remove_known_companions=False,
        # add_radial_regressors=True,
        # include_opposite_regressors=True,
        # annulus_width=5,
    )

    # --- TRAP detection knobs ---
    trap_config.detection = trap_config.detection.merge(
        # detection_threshold=5.0,              # σ threshold in normalized SNR map
        # candidate_threshold=4.75,             # σ threshold to promote to candidate
        # use_spectral_correlation=False,       # DBI has only 2 channels; keep False
        # search_radius=11,                     # px; radius for candidate-clustering
        # inner_mask_radius=1,
        # good_fraction_threshold=0.05,
        # theta_deviation_threshold=25.0,
        # yx_fwhm_ratio_threshold=(1.1, 4.5),
        # save_initial_detection_products=True,
    )

    # --- Per-target stellar parameters used by template matching ---
    # Only used when `config.use_gaia_stellar_parameters` is False, or as the
    # fallback when the observation lacks both Gaia and spectral-type metadata.
    trap_config.detection.stellar_parameters = StellarParameters(
        # teff=8000.0,
        # logg=4.0,
        # feh=0.0,
        # radius=65.0,     # R_sun
        # distance=30.0,   # pc
    )

    # --- TRAP processing knobs ---
    trap_config.processing = trap_config.processing.merge(
        # temporal_components_fraction=[0.2],   # list → loops over each choice
        # wavelength_indices=range(0, 2),       # both DBI channels; index into the wavelength axis
        # verbose=False,                        # more chatty TRAP output when True
        # use_progress_bar=True,
    )

    # Species database directory holds the stellar templates used by TRAP's
    # detection stage. Point this at your local species install.
    species_database_directory = config.directories.base_path / "species"

    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=species_database_directory,
    )


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

"""IFS reduction driver: end-to-end template.

Edit `build_config` and `build_trap_config` for your settings and `TARGET_LIST` for your
targets, then run this file. Other scripts can import the two functions to reuse these
settings; the tutorial run script in docs/tutorials/runs/ does so.
"""
from pathlib import Path

from astropy.table import Table
from trap.parameters import trap_config_for_ifs

from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.cleanup import cleanup_pipeline_products
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations

# List of target names to reduce
TARGET_LIST = ['* bet Pic']

INSTRUMENT = 'ifs'  # Instrument name for the reduction


def build_config(ncpu=4, base_path=Path.home() / "data/sphere"):
    """Reduction settings: CPU count, steps, ESO download, directories, TRAP inputs."""
    # =================== CONFIGURATION ===================
    config = IFSReductionConfig()

    # ===== CONFIGURE CPU RESOURCES (MODIFY THESE TO CHANGE CORE USAGE) =====
    config.set_ncpu(ncpu)  # This sets all CPU parameters to ncpu and applies them

    # ===== CONFIGURE PIPELINE STEPS (MODIFY THESE TO CONTROL WHICH STEPS RUN) =====
    # Convenience methods to enable/disable all IFS steps
    config.steps.disable_all_ifs_steps()
    # Fine-grained control over individual steps
    config.steps = config.steps.merge(
        # Core reduction steps
        download_data=True,
        reduce_calibration=True,
        extract_cubes=True,
        # Bundle settings
        bundle_output=True,
        bundle_hexagons=False,
        bundle_residuals=False,
        compute_frames_info=True,
        cube_header_update=True,
        # Post-processing stepss
        find_centers=True,
        process_extracted_centers=True,
        plot_image_center_evolution=True,
        calibrate_spot_photometry=True,
        calibrate_flux_psf=True,
        spot_to_flux=True,
        # Optional: star-centred copy of the science cube for classical ADI/PCA,
        # tuned via config.alignment. Nothing downstream reads it.
        align_frames=False,
        # TRAP detection steps
        run_trap_reduction=True,
        run_trap_detection=True,
    )

    # ===== RESUME / FORCE =====
    # By default the pipeline RESUMES: any enabled step whose outputs already exist
    # on disk is skipped, so re-running over a growing target list is cheap and
    # adding new targets "just works". To force recomputation:
    #   config.steps = config.steps.merge(force=True)                 # redo everything enabled
    #   config.steps = config.steps.merge(force={"extract_cubes"})    # redo extract_cubes AND all downstream steps (cascade)
    # Forcing align_frames re-runs only itself: it is a leaf step, so it starts no
    # cascade. It is also the only way to regenerate it after changing config.alignment.

    # ===== CONFIGURE ESO DATA DOWNLOAD SETTINGS OF PROPRIETARY DATA =====
    config.preprocessing = config.preprocessing.merge(
        eso_username=None,  # Set to your ESO username if needed
        store_password=False,
        delete_password_after_reduction=False,  # Set to True to remove password from keyring after reduction
    )

    # ===== CONFIGURE DIRECTORY PATHS (MODIFY THESE TO CHANGE LOCATIONS) =====
    config.directories.base_path = Path(base_path)  # Default ~/data/sphere
    config.directories.raw_directory = config.directories.base_path / "data"
    config.directories.reduction_directory = config.directories.base_path / "reduction"

    # ===== OPTIONAL TRAP INPUTS =====
    # Extra per-frame information handed to trap. Values below mirror the config
    # defaults; the two waffle flags are opt-in. Both are gated on continuous-waffle
    # and fall back with an INFO log when the file is absent, so turning them on is
    # inert for observations that cannot produce it.
    config.pass_inverse_variance_to_trap = True                 # ivar cube; improves noise weighting
    config.pass_center_outliers_as_bad_frames_to_trap = False   # True → union of per-channel waffle-fit outliers as trap bad_frames
    config.pass_amplitude_modulation_to_trap = False            # True → loads spot_amplitude_variation.fits

    # Stellar parameters (teff, logg, feh) for template matching are resolved per
    # target: Gaia DR3 (GAIA_TEFF/LOGG/MH) first, then a spectral-type (SP_TYPE)
    # estimate of teff, otherwise the values configured on trap_config in
    # build_trap_config. To force the configured values for all targets:
    # config.use_gaia_stellar_parameters = False
    return config


def build_trap_config(config):
    """TRAP settings for the reduction and the detection, with the CPU budget of `config`."""
    # =================== TRAP CONFIG ===================
    # Create the main TRAP configuration object using the new framework
    trap_config = trap_config_for_ifs()
    # Apply CPU resources from the main config to TRAP
    config.apply_trap_resources(trap_config)

    # ===== CONFIGURE TRAP PARAMETERS (MODIFY THESE TO CHANGE BEHAVIOR) =====
    # Update each sub-config by reassigning `.merge(...)`, which returns a copy with
    # only the named fields overridden (same pattern as `config.steps.merge(...)`
    # above). `trap_config.reduction` is immutable and *must* be updated this way.
    trap_config.reduction = trap_config.reduction.merge(
        search_region_outer_bound=65,  # ~81 pixel is maximum
        # scratch_dir=config.directories.base_path / "scratch",
        # Where TRAP memory-maps the data it shares with its workers. Left unset it
        # picks /dev/shm, which is RAM: a run killed by a signal never reaches the
        # cleanup in its `finally`, and the leaked store keeps consuming memory
        # until you `rm -rf /dev/shm/trap_store_*`. Point it at real disk on shared
        # nodes or under a memory cgroup.
    )
    trap_config.detection = trap_config.detection.merge(
        search_radius=15,  # px; cross-template / cross-channel association radius
        candidate_threshold=4.75,
        detection_threshold=5.0,
        use_spectral_correlation=False,
        # --- candidate search ---
        # The reduction runs down to search_region_inner_bound so the innermost
        # pixels feed the annulus statistics; they are not a detection region. Raise
        # this rather than the inner bound to keep residuals out of the candidate list.
        # minimum_candidate_separation=5.0,   # px
        # candidate_exclusion_radius=None,    # px; None → reuse search_radius. The
        #                                      # exclusion radius already scales with
        #                                      # candidate SNR; set this only to change
        #                                      # the base independently of association.
        # max_candidates=50,                  # cap; each candidate costs a full
        #                                      # contrast-table renormalization
    )
    trap_config.processing = trap_config.processing.merge(
        temporal_components_fraction=[0.15],  # Temporal components fraction
        verbose=False,
        # For surveys or many targets, disabling the progress bar is recommended;
        # progress can be tracked with the reduction_status script.
        use_progress_bar=False,
    )

    # Values used for every target when config.use_gaia_stellar_parameters is False:
    # trap_config.detection.stellar_parameters = trap_config.detection.stellar_parameters.merge(teff=8000.0)
    return trap_config


def select_observations(target_list, database_directory=None, **criteria):
    """Observation rows for the targets after the quality cuts, and their metadata.

    `criteria` are extra `database.filter` keywords, for example `OBS_ID=200363269`.
    """
    # Set $SPHERICAL_DATABASE_DIR to point every entry point at your tables, or pass
    # the directory here explicitly: resolve_database_dir("/path/to/database").
    database_directory = resolve_database_dir(
        database_directory, default=Path.home() / "data/sphere/database")

    # Name of the database files / see Zenodo link in repository for download
    table_of_observations = Table.read(
        database_directory / f"table_of_observations_{INSTRUMENT}.fits")
    table_of_files = Table.read(
        database_directory / f"table_of_files_{INSTRUMENT}.csv")

    database = SphereDatabase(
        table_of_observations, table_of_files, instrument=INSTRUMENT)

    # Select observations for the requested targets and apply quality cuts. Each
    # column is a keyword: a scalar means ==, a list means membership, and a
    # (op, value) tuple applies a comparison ('>', '<', ...), 'in'/'not in', or
    # 'contains'. Rows missing a value for a criterion's column are excluded.
    observation_table = database.filter(
        target_list=target_list,
        TOTAL_EXPTIME_SCI=('>', 30),
        DEROTATOR_MODE='PUPIL',
        HCI_READY=True,
        **criteria,
    )
    # You can select only the first observation that matches the criteria
    # This is useful for testing purposes, you can remove this line to reduce all matching observations
    # observation_table = observation_table[:1]

    observations = database.retrieve_observation_metadata(observation_table)
    return observation_table, observations


# ---------------------Main reduction loop------------------------------------#
def main():
    config = build_config()
    trap_config = build_trap_config(config)
    observation_table, observations = select_observations(TARGET_LIST)
    print(observation_table)

    execute_targets(
        observations=observations,
        config=config)

    # ---------------------TRAP reduction and detection-------------------------#
    # Database and TRAP-specific directories (not part of spherical IFS reduction)
    species_database_directory = Path(config.directories.base_path) / "species"
    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=species_database_directory,
    )
    return observations, config


def cleanup(observations_list, config_obj, dry_run=True, clean_raw=False, clean_extracted=True, clean_wavecal=True):
    """
    Wrapper function for cleanup_pipeline_products with convenient defaults.

    IMPORTANT: Only run cleanup functions after verifying that cube building
    completed successfully and you have verified the final data products.

    Parameters:
    -----------
    observations_list : list
        List of observations to clean, as returned by `main()`
    config_obj : object
        Configuration object, as returned by `main()`
    dry_run : bool, default True
        If True, only show what would be cleaned without actually deleting files
    clean_raw : bool, default False
        Whether to clean raw data files (keep False by default for safety)
    clean_extracted : bool, default True
        Whether to clean intermediate extracted cube files
    clean_wavecal : bool, default True
        Whether to clean wavelength calibration files
    """
    cleanup_pipeline_products(
        observations_list=observations_list,
        config_obj=config_obj,
        dry_run=dry_run,
        clean_raw=clean_raw,
        clean_extracted=clean_extracted,
        clean_wavecal=clean_wavecal
    )

if __name__ == "__main__":
    observations, config = main()

    # Examples of cleanup usage:
    # Dry run (safe, shows what would be cleaned):
    # cleanup(observations, config)

    # Actually clean files (remove dry_run=False to execute):
    # cleanup(observations, config, dry_run=False)

    # Custom cleanup with specific parameters:
    # cleanup(observations, config, dry_run=False, clean_raw=False, clean_extracted=True, clean_wavecal=True)

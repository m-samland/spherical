from pathlib import Path

from astropy.table import Table
from trap.parameters import trap_config_for_ifs

from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.cleanup import cleanup_pipeline_products
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations

# List of target names to reduce
target_list = ['* bet Pic']

check_cubebuilding_output = False

# =================== CONFIGURATION ===================
config = IFSReductionConfig()

# ===== CONFIGURE CPU RESOURCES (MODIFY THESE TO CHANGE CORE USAGE) =====
config.set_ncpu(4)  # This sets all CPU parameters to 4 and applies them

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
    # TRAP detection steps
    run_trap_reduction=True,
    run_trap_detection=True,
)

# ===== CONFIGURE ESO DATA DOWNLOAD SETTINGS OF PROPRIETARY DATA =====
config.preprocessing = config.preprocessing.merge(
    eso_username=None,  # Set to your ESO username if needed
    store_password=False,
    delete_password_after_reduction=False,  # Set to True to remove password from keyring after reduction
)

# ===== CONFIGURE DIRECTORY PATHS (MODIFY THESE TO CHANGE LOCATIONS) =====
config.directories.base_path = Path.home() / "data/sphere"  # Default path
config.directories.raw_directory = config.directories.base_path / "data" 
config.directories.reduction_directory = config.directories.base_path / "reduction"

# Database and TRAP-specific directories (not part of spherical IFS reduction)
database_directory = Path.home() / "data/sphere/database"
species_database_directory = Path(config.directories.base_path) / "species"

instrument = 'ifs'  # Instrument name for the reduction

# Name of the database files / see Zenodo link in repository for download
table_of_observations = Table.read(
    database_directory / f"table_of_observations_{instrument}.fits")
table_of_files = Table.read(
    database_directory / f"table_of_files_{instrument}.csv")

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
)
trap_config.detection = trap_config.detection.merge(
    search_radius=15,  # Exclusion radius around candidates (pixel); larger for bright companions to avoid contamination
    candidate_threshold=4.75,
    detection_threshold=5.0,
    use_spectral_correlation=False,
)
trap_config.processing = trap_config.processing.merge(
    temporal_components_fraction=[0.15],  # Temporal components fraction
    verbose=False,
    # For surveys or many targets, disabling the progress bar is recommended;
    # progress can be tracked with the reduction_status script.
    use_progress_bar=False,
)

# Stellar parameters (teff, logg, feh) for template matching are resolved per
# target: Gaia DR3 (GAIA_TEFF/LOGG/MH) first, then a spectral-type (SP_TYPE)
# estimate of teff, otherwise the values configured on trap_config below.
# To force the configured values for all targets, disable the lookup and set them:
# config.use_gaia_stellar_parameters = False
# trap_config.detection.stellar_parameters = trap_config.detection.stellar_parameters.merge(teff=8000.0)

# ---------------------Database setup-----------------------------------------#
# Modify this section to select which data you want to download and reduce
database = SphereDatabase(
    table_of_observations, table_of_files, instrument=instrument)

# Select observations for the requested targets and apply quality cuts. Each
# column is a keyword: a scalar means ==, a list means membership, and a
# (op, value) tuple applies a comparison ('>', '<', ...), 'in'/'not in', or
# 'contains'. Rows missing a value for a criterion's column are excluded.
observation_table = database.filter(
    target_list=target_list,
    TOTAL_EXPTIME_SCI=('>', 30),
    DEROTATOR_MODE='PUPIL',
    HCI_READY=True,
)
# You can select only the first observation that matches the criteria
# This is useful for testing purposes, you can remove this line to reduce all matching observations
# observation_table = observation_table[:1]
print(observation_table)

observations = database.retrieve_observation_metadata(observation_table)

# ---------------------Main reduction loop------------------------------------#
def main():
    execute_targets(
        observations=observations,
        config=config)
    
    # ---------------------TRAP reduction and detection-------------------------#
    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=species_database_directory,
    )


def cleanup(dry_run=True, clean_raw=False, clean_extracted=True, clean_wavecal=True, observations_list=None, config_obj=None):
    """
    Wrapper function for cleanup_pipeline_products with convenient defaults.
    
    IMPORTANT: Only run cleanup functions after verifying that cube building
    completed successfully and you have verified the final data products.
    
    Parameters:
    -----------
    dry_run : bool, default True
        If True, only show what would be cleaned without actually deleting files
    clean_raw : bool, default False
        Whether to clean raw data files (keep False by default for safety)
    clean_extracted : bool, default True
        Whether to clean intermediate extracted cube files
    clean_wavecal : bool, default True
        Whether to clean wavelength calibration files
    observations_list : list, optional
        List of observations to clean. If None, uses global 'observations'
    config_obj : object, optional
        Configuration object. If None, uses global 'config'
    """
    # Use provided parameters or fall back to globals
    obs_list = observations_list if observations_list is not None else observations
    cfg = config_obj if config_obj is not None else config
    
    # Call the main cleanup function from the module
    cleanup_pipeline_products(
        observations_list=obs_list,
        config_obj=cfg,
        dry_run=dry_run,
        clean_raw=clean_raw,
        clean_extracted=clean_extracted,
        clean_wavecal=clean_wavecal
    )

if __name__ == "__main__":
    main()
    
    # Examples of cleanup usage:
    # Dry run (safe, shows what would be cleaned):
    # cleanup()
    
    # Actually clean files (remove dry_run=False to execute):
    # cleanup(dry_run=False)
    
    # Custom cleanup with specific parameters:
    # cleanup(dry_run=False, clean_raw=False, clean_extracted=True, clean_wavecal=True)
    
    # One-liner after successful reduction:
    # cleanup(dry_run=False, clean_extracted=True, clean_wavecal=True)
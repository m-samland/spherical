from pathlib import Path

import numpy as np
from astropy.table import Table
from trap.parameters import trap_config_for_ifs

from spherical.database.sphere_database import SphereDatabase
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
    # Post-processing stepss
    compute_frames_info=True,
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
trap_config.reduction.search_region_outer_bound = 81
trap_config.processing.temporal_components_fraction = [0.15]  # Temporal components fraction
trap_config.detection.search_radius = 15 # Exclusion radius around candidates in pixel, bigger for bright companions to avoid contamination

# Configure detection parameters
trap_config.detection.candidate_threshold = 4.75
trap_config.detection.detection_threshold = 5.0
trap_config.detection.use_spectral_correlation = False
trap_config.processing.verbose = False

# Configure stellar parameters for template matching, e.g.:
trap_config.detection.stellar_parameters.teff = 8000.0

# ---------------------Database setup-----------------------------------------#
# Modify this section to select which data you want to download and reduce
database = SphereDatabase(
    table_of_observations, table_of_files, instrument=instrument)

observation_table = database.target_list_to_observation_table(target_list)
# Apply filters to select observations
observation_table_mask = np.logical_and.reduce([
    observation_table['TOTAL_EXPTIME_SCI'] > 30,
    observation_table['DEROTATOR_MODE'] == 'PUPIL',
    observation_table['HCI_READY'] == True,]
)
# Another useful keyword is 'OBS_PROG_ID', the program ID of the survey you want to reduce. 

observation_table = observation_table[observation_table_mask]
# IMPORTANT: We select only the first observation that matches the criteria
# This is useful for testing purposes, you can remove this line to reduce all matching observations
observation_table = observation_table[:1]
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

if __name__ == "__main__":
    main()
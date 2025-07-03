from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from trap.parameters import trap_config_for_ifs

from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline import ifs_reduction
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations

# List of target names to reduce, e.g. ['51 Eri', 'Beta Pic']
target_list = ['* bet Pic']

# Post-processing / Exoplanet detection
run_trap_reduction = True
run_trap_detection = True
overwrite_trap = True
check_cubebuilding_output = False

ncpu_trap = 100

# =================== NEW CONFIGURATION FRAMEWORK ===================
# Create the main configuration object using the new framework
config = IFSReductionConfig()

# ===== CONFIGURE CPU RESOURCES (MODIFY THESE TO CHANGE CORE USAGE) =====
# Configure resource settings (CPU cores) - users can modify these values
config.resources.ncpu_calib = 100        # CPU cores for calibration
config.resources.ncpu_extract = 100      # CPU cores for cube extraction  
config.resources.ncpu_center = 100       # CPU cores for center finding

# Apply resource settings to all sub-configurations
# This ensures consistent CPU usage across all pipeline steps
config.apply_resources()

# ===== CONFIGURE PIPELINE STEPS (MODIFY THESE TO CONTROL WHICH STEPS RUN) =====
# Configure which parts of the pipeline to run
config.steps = config.steps.merge(
    # Core reduction steps
    download_data=False,
    reduce_calibration=False,
    extract_cubes=False,
    
    # Bundle settings
    bundle_output=False,
    bundle_hexagons=False,
    bundle_residuals=False,
    
    # Post-processing steps
    compute_frames_info=True,
    find_centers=False,
    process_extracted_centers=False,
    plot_image_center_evolution=False,
    calibrate_spot_photometry=False,
    calibrate_flux_psf=False,
    spot_to_flux=False,
    
    # Overwrite settings
    overwrite_calibration=True,
    overwrite_bundle=True,
    overwrite_preprocessing=True,
)

# ===== CONFIGURE ESO DATA DOWNLOAD SETTINGS =====
# Configure ESO data download settings - users can modify these
config.preprocessing = config.preprocessing.merge(
    eso_username=None,  # Set to your ESO username if needed
    store_password=False,
    delete_password_after_reduction=False,  # Set to True to remove password after reduction
)

# ===== CONFIGURE DIRECTORY PATHS (MODIFY THESE TO CHANGE LOCATIONS) =====
# Configure directory settings - users can modify these paths
# config.directories.base_path = Path("/custom/sphere/")
# config.directories.raw_directory = Path("/custom/sphere/data/") 
# config.directories.reduction_directory = Path("custom/sphere/rreduction")
config.directories.raw_directory = config.directories.base_path / "data_test" 
config.directories.reduction_directory = config.directories.base_path / "reduction_test" #Path("/custom/reduction/output")
# ===== END DIRECTORY CONFIGURATION =====

# Database and TRAP-specific directories (not part of spherical IFS reduction)
# database_directory = Path(config.directories.base_path) / "database"
database_directory = Path.home() / "data/sphere/database"
species_database_directory = Path(config.directories.base_path) / "species"

instrument = 'ifs'  # Instrument name for the reduction

# Name of the database files / see files in Git repository
table_of_observations = Table.read(
    database_directory / f"table_of_observations_{instrument}.fits")
table_of_files = Table.read(
    database_directory / f"table_of_files_{instrument}.csv")

# =================== END NEW CONFIGURATION ===================

# =================== NEW TRAP CONFIGURATION FRAMEWORK ===================
# Create the main TRAP configuration object using the new framework
trap_config = trap_config_for_ifs()

# ===== CONFIGURE TRAP PARAMETERS (MODIFY THESE TO CHANGE BEHAVIOR) =====
trap_config.processing.temporal_components_fraction = [0.20]  # Temporal components fraction
trap_config.processing.overwrite_reduction = overwrite_trap

# Configure CPU resources for TRAP
trap_config.resources.ncpu_reduction = ncpu_trap
trap_config.resources.ncpu_detection = ncpu_trap

# Apply resource settings to all sub-configurations
trap_config.apply_resources()

# Configure detection parameters
trap_config.detection.candidate_threshold = 4.75
trap_config.detection.detection_threshold = 5.0
trap_config.detection.use_spectral_correlation = False

# Configure stellar parameters for template matching
trap_config.detection.stellar_parameters.teff = 8000.0

# ===== END TRAP CONFIGURATION =====

# =================== END NEW TRAP CONFIGURATION ===================

# ---------------------Database setup-----------------------------------------#
# Modify this section to select which data you want to download and reduce
database = SphereDatabase(
    table_of_observations, table_of_files, instrument=f'{instrument}')

obs_table = []
for target_name in target_list:
    obs_table.append(database.get_observation_SIMBAD(target_name))

obs_table = vstack(obs_table)
# obs_table = unique(obs_table, keys=['MAIN_ID', 'FILTER', 'NIGHT_START'])
# obs_table = table_of_observations
# Filter which observations to reduce
obs_table_mask = np.logical_and.reduce([
    obs_table['TOTAL_EXPTIME_SCI'] > 30,
    obs_table['DEROTATOR_MODE'] == 'PUPIL',
    obs_table['HCI_READY'] == True,]
)

# # Example: only reduce the first available data set
obs_table = obs_table[obs_table_mask][:1]
print(obs_table)

observations = database.retrieve_observation_metadata(obs_table)

# ---------------------Main reduction loop------------------------------------#
def main():
    ifs_reduction.execute_targets(
        observations=observations,
        config=config)
    
    # ---------------------TRAP reduction and detection-------------------------#
    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=species_database_directory,
        run_trap_reduction=run_trap_reduction,
        run_trap_detection=run_trap_detection,
    )

if __name__ == "__main__":
    main()
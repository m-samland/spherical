import os
from copy import copy
from pathlib import Path

import keyring
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astroquery.eso import Eso
from trap.detection import DetectionAnalysis
from trap.parameters import Instrument, Reduction_parameters
from trap.reduction_wrapper import run_complete_reduction

from spherical.database.sphere_database import Sphere_database
from spherical.pipeline import ifs_reduction
from spherical.pipeline.toolbox import make_target_folder_string

# List of target names to reduce, e.g. ['51 Eri', 'Beta Pic']
target_list = ['']

# IFS Basic Steps
download_data = False
reduce_calibration = False
run_cubebuilding = False
check_cubebuilding_output = False

# Overwrite output settings
overwrite_calibration = False
overwrite_bundle = False
overwrite_preprocessing = True

# IFS Pre-reduction Steps
bundle_output = False
bundle_hexagons = False
bundle_residuals = False
compute_frames_info = False
find_centers = False
plot_image_center_evolution = False
process_extracted_centers = False
calibrate_spot_photometry = False
calibrate_flux_psf = False
spot_to_flux = False

# Post-processing / Exoplanet detection
run_trap_reduction = False
run_trap_detection = False
overwrite_trap = False

# Multiprocessing settings
ncpu_calibration = 4
ncpu_cubebuilding = 4
ncpu_find_center = 4
ncpu_trap = 4

# Directory settings
base_path = Path.home() / "data/sphere"

database_directory = base_path / "data/sphere/database"
raw_directory = base_path / "data"
reduction_directory = base_path / "reduction_test"
# Directory to save the species-package database for TRAP spectral template matching
species_database_directory = base_path / "species"

# ESO data download settings
eso_username = None
store_password = False
delete_password_after = False

# Name of the database files / see files in Git repository
table_of_observations = Table.read(
    database_directory / "table_of_IFS_observations.fits")
table_of_files = Table.read(
    database_directory / "table_of_IFS_files.csv")

# IFS pipeline calibration parameters
calibration_step_parameters = {
    'mask': None,  # use standard mask
    'order': None,  # use standard polynomial order
    'upsample': True,
    'ncpus': ncpu_calibration,
    'verbose': True}

# IFS pipeline cube extraction parameters
cube_extraction_parameters = {
    'individual_dits': True,
    'noisefac': 0.05,
    'gain': 1.8,
    'saveramp': False,
    'bgsub': True,
    'flatfield': True,
    'mask': True,
    'method': 'optext',
    'fitshift': True,
    'suppressrn': False,
    'minpct': 70,
    'refine': True,
    'crosstalk_scale': 1.,
    'dc_xtalk_correction': False,
    'linear_wavelength': True,
    'fitbkgnd': False,
    'smoothandmask': True,
    'resample': True,
    'saveresid': True,
    'verbose': True
}

# IFS pipeline generic pre-processing parameters
preprocessing_parameters = {
    'ncpu_cubebuilding': ncpu_cubebuilding,
    'bg_pca': True,
    'subtract_coro_from_center': False,
    'exclude_first_flux_frame': True,
    'exclude_first_flux_frame_all': True,
    'flux_combination_method': 'median',
    'ncpu_find_center': ncpu_find_center,
}

# TRAP parameters
# Wavelength indices to reduce, first and last frame are skipped due to low S/N
wavelength_indices = np.arange(1, 38)
# Number of temporal principal components to fit
temporal_components_fraction = [0.2] 

trap_parameters = Reduction_parameters(
    search_region_inner_bound=1,
    search_region_outer_bound=81,
    data_auto_crop=True,
    data_crop_size=None,
    right_handed=False,
    include_noise=False,
    temporal_model=True,
    temporal_plus_spatial_model=False,
    second_stage_trap=False,
    remove_model_from_spatial_training=True,
    remove_bad_residuals_for_spatial_model=True,
    spatial_model=False,
    local_temporal_model=False,
    local_spatial_model=False,
    protection_angle=0.5,
    spatial_components_fraction=0.3,
    spatial_components_fraction_after_trap=0.1,
    yx_known_companion_position=None,
    known_companion_contrast=None,
    use_multiprocess=True,
    ncpus=ncpu_trap,
    # Reduction and signal masks
    autosize_masks_in_lambda_over_d=True,
    reduction_mask_size_in_lambda_over_d=1.1,
    signal_mask_size_in_lambda_over_d=2.1,
    reduction_mask_psf_size=19,
    signal_mask_psf_size=19,
    # Regressor selection
    annulus_width=5,
    add_radial_regressors=True,
    include_opposite_regressors=True,
    # Contrast curve / normalization stuff
    contrast_curve=True,
    contrast_curve_sigma=5.,
    normalization_width=3,
    companion_mask_radius=11)

# TRAP detection settings
candidate_threshold = 4.75
detection_threshold = 5.0
use_spectral_correlation = False

# Stellar parameters used for host star by TRAP template matching
stellar_parameters = {
    "teff": 7800,
    "logg": 3.5,
    "feh": 0.0,
    "radius": 65.0,
    "distance": 30.0,
}

# ---------------------Database setup-----------------------------------------#
database = Sphere_database(
    table_of_observations, table_of_files, instrument='IFS')

eso = Eso()
# Unfortunately we need to store the password to allow login in for each data set
# Otherwise the login credentials will time out if you reduce many data sets
# You can delete your password with the keyring after using this code if you need
if eso_username is not None:
    eso.login(username=eso_username, store_password=store_password)

obs_table = []
for target_name in target_list:
    obs_table.append(database.get_observation_SIMBAD(target_name))

obs_table = vstack(obs_table)

# Filter which observations to reduce
obs_table_mask = np.logical_and.reduce([
    obs_table['TOTAL_EXPTIME'] > 30,
    obs_table['DEROTATOR_MODE'] == 'PUPIL',
    obs_table['FAILED_SEQ'] == False])

obs_table = obs_table[obs_table_mask]
print(obs_table)

observation_object_list = database.retrieve_observation_object_list(obs_table)

# ---------------------Main reduction loop------------------------------------#
def main():
    for observation in observation_object_list:
        frame_types_to_extract = []
        if len(observation.frames['CORO']) > 0:
            frame_types_to_extract.append('CORO')
        if len(observation.frames['CENTER']) > 0:
            frame_types_to_extract.append('CENTER')
        if len(observation.frames['FLUX']) > 0:
            frame_types_to_extract.append('FLUX')
        
        ifs_reduction.execute_IFS_target(
            observation=observation,
            calibration_parameters=calibration_step_parameters,
            extraction_parameters=cube_extraction_parameters,
            reduction_parameters=preprocessing_parameters,
            reduction_directory=reduction_directory,
            raw_directory=raw_directory,
            download_data=download_data,
            eso_username=eso_username,
            reduce_calibration=reduce_calibration,
            extract_cubes=run_cubebuilding,
            frame_types_to_extract=frame_types_to_extract,
            bundle_output=bundle_output,
            bundle_hexagons=bundle_hexagons,
            bundle_residuals=bundle_residuals,
            compute_frames_info=compute_frames_info,
            find_centers=find_centers,
            plot_image_center_evolution=plot_image_center_evolution,
            process_extracted_centers=process_extracted_centers,
            calibrate_spot_photometry=calibrate_spot_photometry,
            calibrate_flux_psf=calibrate_flux_psf,
            spot_to_flux=spot_to_flux,
            overwrite_calibration=overwrite_calibration,
            overwrite_bundle=overwrite_bundle,
            overwrite_preprocessing=overwrite_preprocessing,
            save_plots=True,
            verbose=cube_extraction_parameters['verbose'])
        print('Finished reduction for observation {}.'.format(observation))
    
    if eso_username is not None and delete_password_after:
        keyring.delete_password('astroquery:www.eso.org', eso_username)

    if check_cubebuilding_output:
        reduced, missing_files = ifs_reduction.check_output(reduction_directory, observation_object_list)
        print(reduced)
        print(missing_files)

    for observation in observation_object_list:
        if not run_trap_reduction and not run_trap_detection:
            print("No reduction selected to run. Exiting.")
            break
        
        obs_mode = observation.observation['IFS_MODE'][0]
        assert obs_mode in ['OBS_YJ', 'OBS_H'], "Observation has to be done with IFS."

        if obs_mode == 'OBS_YJ':
            spectral_resolution = 55
        elif obs_mode == 'OBS_H':
            spectral_resolution = 35

        continuous_satellite_spots = observation.observation['WAFFLE_MODE'][0]

        used_instrument = Instrument(
            name="IFS",
            pixel_scale=u.pixel_scale(0.00746 * u.arcsec / u.pixel),
            telescope_diameter=7.99 * u.m,
            detector_gain=1.0,
            readnoise=0.0,
            instrument_type="ifu",
            wavelengths=None,
            spectral_resolution=spectral_resolution,
            filters=None,
            transmission=None,
        )

        data_directory = ifs_reduction.output_directory_path(
            reduction_directory,
            observation,
            method=cube_extraction_parameters['method'])
        
        name_mode_date = make_target_folder_string(observation)
        result_folder = os.path.join(reduction_directory, 'IFS/trap', name_mode_date)
        trap_parameters.result_folder = result_folder

        if continuous_satellite_spots:
            file_identifier = "center"
        else:
            file_identifier = "coro"

        wavelengths = (
            fits.getdata(os.path.join(data_directory, "wavelengths.fits")) * u.nm
        ).to(u.micron)
        used_instrument.wavelengths = wavelengths

        pa = pd.read_csv(
            os.path.join(data_directory, f"frames_info_{file_identifier}.csv")
        )['DEROT ANGLE'].values

        data_full = fits.getdata(
            os.path.join(data_directory, f"{file_identifier}_cube.fits")
        )
        flux_psf_full = fits.getdata(
            os.path.join(data_directory, "flux_stamps_calibrated_bg_corrected.fits")
        )
        flux_psf_full = np.mean(flux_psf_full, axis=1)
        # flux_psf_full = flux_psf_full[:, 0]

        xy_image_centers = fits.getdata(
            os.path.join(data_directory, "image_centers_fitted_robust.fits")
        )
        if not continuous_satellite_spots:
            xy_image_centers = np.nanmean(xy_image_centers, axis=1)
            xy_image_centers = xy_image_centers[:, None, :].repeat(len(pa), axis=1)

        # Waffle amplitudes
        amplitude_modulation_full = None
        inverse_variance_full = None
        bad_pixel_mask_full = None
        bad_frames = None

        if run_trap_reduction:
            _ = run_complete_reduction(
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                instrument=used_instrument,
                reduction_parameters=copy(trap_parameters),
                temporal_components_fraction=temporal_components_fraction,
                wavelength_indices=wavelength_indices,
                inverse_variance_full=inverse_variance_full,
                bad_frames=bad_frames,
                amplitude_modulation_full=amplitude_modulation_full,
                xy_image_centers=xy_image_centers,
                overwrite=overwrite_trap,
                verbose=False,
            )

        if run_trap_detection:
            analysis = DetectionAnalysis(
                reduction_parameters=trap_parameters, instrument=used_instrument
            )
            analysis.read_output(
                temporal_components_fraction[0],
                result_folder=trap_parameters.result_folder,
                reduction_type="temporal",
                correlated_residuals=False,
                read_parameters=True,
            )

            analysis.reduction_parameters.result_folder = result_folder

            analysis.detection_and_characterization_with_template_matching(
                reduction_parameters=copy(trap_parameters),
                instrument=used_instrument, 
                species_database_directory=species_database_directory,
                stellar_parameters=stellar_parameters,
                data_full=data_full,
                flux_psf_full=flux_psf_full,
                pa=pa,
                temporal_components_fraction=temporal_components_fraction[0],
                wavelength_indices=wavelength_indices,
                xy_image_centers=xy_image_centers, 
                inverse_variance_full=inverse_variance_full,
                bad_frames=bad_frames,
                bad_pixel_mask_full=bad_pixel_mask_full, 
                amplitude_modulation_full=amplitude_modulation_full, 
                detection_threshold=detection_threshold,
                candidate_threshold=candidate_threshold,
                use_spectral_correlation=use_spectral_correlation,
            )

if __name__ == "__main__":
    main()
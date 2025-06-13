import os
from glob import glob

import numpy as np
from astropy.io import fits
from natsort import natsorted


def bundle_IFS_output_into_cubes(key, cube_outputdir, output_type='resampled', overwrite=True):
    """
    Assemble individual IFS data cubes and associated metadata into 4D master cubes for a given frame type.

    This function collects all extracted IFS data cubes (e.g., for 'CORO', 'CENTER', or 'FLUX' frames) from a specified directory,
    stacks them along the temporal axis, and writes out combined data, inverse variance, and parallactic angle arrays as FITS files.
    It supports different output types (resampled, hexagons, residuals) and is typically called as the final step of the IFS pipeline
    for each frame type.

    Parameters
    ----------
    key : str
        Frame type to bundle (e.g., 'CORO', 'CENTER', 'FLUX').
    cube_outputdir : str
        Path to the directory containing extracted cubes for this frame type.
    output_type : {'resampled', 'hexagons', 'residuals'}, optional
        Which type of cube to bundle. 'resampled' (default) for wavelength-resampled cubes,
        'hexagons' for native hexagonal grid cubes, 'residuals' for residual cubes.
    overwrite : bool, optional
        If True (default), overwrite existing output FITS files.

    Returns
    -------
    None
        The function writes output FITS files to disk and returns None.

    Notes
    -----
    - The output cubes have shape (n_wave, n_time, ny, nx), with axes swapped for compatibility with downstream analysis.
    - Parallactic angles are extracted from the FITS headers if available and saved as a separate FITS file.
    - This function is called automatically by the pipeline after cube extraction, but can also be run manually.

    Examples
    --------
    >>> bundle_IFS_output_into_cubes('CORO', '/path/to/output', output_type='resampled')
    >>> bundle_IFS_output_into_cubes('CENTER', '/path/to/output', output_type='hexagons', overwrite=False)
    """

    extracted_dir = os.path.join(cube_outputdir, key) + '/'
    converted_dir = os.path.join(cube_outputdir, 'converted') + '/'
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)

    if output_type == 'resampled':
        glob_pattern = 'SPHER.*cube_resampled_DIT*.fits'
        name_suffix = ''
    elif output_type == 'hexagons':
        glob_pattern = 'SPHER.*cube_DIT*.fits'
        name_suffix = 'hexagons_'
    elif output_type == 'residuals':
        glob_pattern = 'SPHER.*cube_residuals_DIT*.fits'
        name_suffix = 'residuals_'
    else:
        raise ValueError('Invalid output_type selected.')

    science_files = natsorted(
        glob(os.path.join(extracted_dir, glob_pattern), recursive=False))

    if len(science_files) == 0:
        print('No output found in: {}'.format(extracted_dir))
        return None

    data_cube = []
    inverse_variance_cube = []

    for file in science_files:
        hdus = fits.open(file)
        data_cube.append(hdus[1].data.astype('float32'))
        inverse_variance_cube.append(hdus[2].data.astype('float32'))
        hdus.close()

    data_cube = np.array(data_cube, dtype='float32')
    inverse_variance_cube = np.array(inverse_variance_cube, dtype='float32')

    data_cube = np.swapaxes(data_cube, 0, 1)
    inverse_variance_cube = np.swapaxes(inverse_variance_cube, 0, 1)

    fits.writeto(os.path.join(converted_dir, '{}_{}cube.fits'.format(
        key, name_suffix).lower()), data_cube.astype('float32'), overwrite=overwrite)
    fits.writeto(os.path.join(converted_dir, '{}_{}ivar_cube.fits'.format(
        key, name_suffix).lower()), inverse_variance_cube.astype('float32'), overwrite=overwrite)


def run_bundle_output(
    frame_types_to_extract,
    cube_outputdir,
    converted_dir,
    extraction_parameters,
    instrument,
    non_least_square_methods,
    overwrite_bundle,
    bundle_hexagons,
    bundle_residuals
):
    """
    Bundle IFS output cubes and write wavelength solution to output directory.

    Parameters
    ----------
    frame_types_to_extract : list of str
        Frame types to bundle (e.g., ["FLUX", "CENTER", "CORO"])
    cube_outputdir : str
        Directory containing extracted cubes.
    converted_dir : str
        Directory to write converted/bundled outputs.
    extraction_parameters : dict
        Extraction parameters, must include 'method' and 'linear_wavelength'.
    instrument : object
        Instrument object with wavelength_range and lam_midpts attributes.
    non_least_square_methods : list of str
        List of extraction methods that are not least-squares.
    overwrite_bundle : bool
        Whether to overwrite existing bundle outputs.
    bundle_hexagons : bool
        Whether to bundle hexagon outputs.
    bundle_residuals : bool
        Whether to bundle residual outputs.
    """

    for key in frame_types_to_extract:
        if bundle_hexagons:
            bundle_IFS_output_into_cubes(
                key, cube_outputdir, output_type='hexagons', overwrite=overwrite_bundle)
        if bundle_residuals:
            bundle_IFS_output_into_cubes(
                key, cube_outputdir, output_type='residuals', overwrite=overwrite_bundle)
        bundle_IFS_output_into_cubes(
            key, cube_outputdir, output_type='resampled', overwrite=overwrite_bundle)

    if (extraction_parameters['method'] in non_least_square_methods) \
            and extraction_parameters['linear_wavelength']:
        wavelengths = np.linspace(
            instrument.wavelength_range[0].value,
            instrument.wavelength_range[1].value,
            39
        )
    else:
        wavelengths = instrument.lam_midpts

    fits.writeto(os.path.join(converted_dir, 'wavelengths.fits'),
                 wavelengths, overwrite=overwrite_bundle)

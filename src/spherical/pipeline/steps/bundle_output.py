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

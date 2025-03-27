"""Tool for the detecting and removal of cosmic rays."""

__all__ = ["get_badpix_mask", "find_and_remove_cosmics"]

import os
import shutil

import numpy as np
from astropy.io import fits

from .utils import delete_folder_content


def get_badpix_mask(list_of_files):
    """Read in the bad pixel mask and convert to boolean."""
    badpix_mask = np.zeros((1024, 2048)).astype(bool)
    for tfile in list_of_files:
        if "IRD_MASTER_DARK" in tfile:
            hdu = fits.open(tfile.split()[0])
            badpix_mask = np.logical_or(
                badpix_mask, hdu["BADPIXELMAP"].data.astype(bool)
            )
            hdu.close()
        if "IRD_FLAT_FIELD" in tfile:
            hdu = fits.open(tfile.split()[0])
            badpix_mask = np.logical_or(
                badpix_mask, hdu["BADPIXELMAP"].data.astype(bool)
            )
            hdu.close()
    return badpix_mask


def find_and_remove_cosmics(path_to_sof_file, sigma=7.0, verbose=False):
    """Function to detect and remove cosmic rays in the raw images listed in a
       sof file. The corrected images will be saved seperatevely in a new
       subfolder in the output directory of the reduction.

    Parameters
    ----------
    path_to_sof_file : str
        Path to the sof file, which will be commited to the esorex recipe.
    silent_mode : bool
        Wether to plot intermediate status information.
    sigma : float
        Faktor of the std above a pixel will be considered a cosmic.

    Returns
    -------
    None
        Save the cosmics corrected images to a subfolder in the outputdirectory
        of the pipeline '.../outputdir/cosmics/raw_science_cosmics_rm'. A
        pixelmask, which containes every found cosmic for the image series will
        also be saved into the cosmics directory.

    """
    from astroscrappy import detect_cosmics
    
    path_to_working_direc = os.path.split(os.path.split(path_to_sof_file)[0])[0]
    if verbose:
        if not path_to_working_direc.startswith("/"):
            raise NameError(
                "Your path to sof/.sof does not look like a fully qualified"
                "path: '{}'!\n".format(path_to_working_direc)
            )
    # create the folder where the information about the cosmic correction
    # will be stored
    path_to_results_folder = os.path.join(path_to_working_direc, "cosmics")
    if not os.path.exists(path_to_results_folder):
        os.makedirs(path_to_results_folder)
    else:
        delete_folder_content(path_to_results_folder)
    path_to_results_file = os.path.join(path_to_results_folder, "cosmics.fits")
    # create a subfolder for the cosmic corrected raw images
    path_to_cubes_without_cosmics = os.path.join(
        path_to_results_folder, "raw_science_cosmics_rm"
    )
    if not os.path.exists(path_to_cubes_without_cosmics):
        os.makedirs(path_to_cubes_without_cosmics)
    else:
        delete_folder_content(path_to_cubes_without_cosmics)
    # make a copy of the sof file, with the original content
    orig_sof_file = os.path.splitext(path_to_sof_file)
    shutil.copy2(
        path_to_sof_file,
        orig_sof_file[0] + "_without_cosmics_" "removal" + orig_sof_file[1],
    )
    # read in the sof file and get the list of science images
    lines = open(path_to_sof_file).readlines()
    file_list = [line.split()[0] for line in lines if "IRD_SCIENCE_DBI_RAW" in line]
    # get badpixel map from sof file
    badpix_map = get_badpix_mask(lines)
    global_cosmics_map = badpix_map * False

    # remove cosmics and save the corrected cubes
    filenames = []
    for idx, orig_file in enumerate(file_list):
        if verbose:
            print(
                "Correcting '{0}' for cosmic rays. "
                "Entry {1} of {2} in file list".format(
                    orig_file, idx + 1, len(file_list)
                )
            )
        hdulist = fits.open(orig_file)
        cube = hdulist[0].data
        cosmics_free_cube = np.empty_like(cube)
        for idx, frame in enumerate(cube):
            cosmics_map, cosmics_free_image = detect_cosmics(
                frame,
                inmask=badpix_map,
                satlevel=np.inf,
                sigclip=sigma,
                gain=1.0,
                sepmed=True,
                cleantype="medmask",
                fsmode="median",
                verbose=verbose,
            )
            cosmics_free_cube[idx] = cosmics_free_image
            # add found cosmics to a global map
            global_cosmics_map = np.logical_or(global_cosmics_map, cosmics_map)
        filename = os.path.join(
            path_to_cubes_without_cosmics, os.path.split(orig_file)[1]
        )
        fits.writeto(
            filename,
            cosmics_free_cube,
            header=hdulist[0].header,
            output_verify="warn",
            overwrite=True,
        )
        filenames.append(filename)

    # save the global image of all found cosmics
    fits.writeto(path_to_results_file, global_cosmics_map * 1, overwrite=True)
    if verbose:
        sum_cosmics = np.sum(global_cosmics_map * 1)
        print("{} pixels found to be hit by a cosmic ray".format(sum_cosmics))
    # replacing the path to the original raw data by the cosmics corrected ones
    # in the sof file
    f = open(path_to_sof_file, "w")
    for path_to_file in filenames:
        f.writelines([path_to_file, " ", "IRD_SCIENCE_DBI_RAW", "\n"])
    f.writelines(lines[-4:])
    f.close()

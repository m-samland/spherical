import os
from glob import glob

import charis


def run_wavelength_calibration(
    observation,
    instrument,
    calibration_parameters,
    wavecal_outputdir,
    overwrite_calibration=False
):
    """
    Run the wavelength calibration step for the IFS pipeline.

    Parameters
    ----------
    observation : object
        Observation object containing frame info, must have a 'frames' attribute with 'WAVECAL' key.
    instrument : object
        Instrument object (e.g., charis.instruments.SPHERE), must have a 'calibration_wavelength' attribute.
    calibration_parameters : dict
        Dictionary of calibration parameters, must include 'ncpus' key for parallelization.
    wavecal_outputdir : str
        Output directory for wavelength calibration products.
    overwrite_calibration : bool, optional
        If True, overwrite existing calibration products. Default is False.

    Returns
    -------
    None
        This function performs calibration as a side effect and does not return a value.

    Notes
    -----
    - Creates output directories if they do not exist.
    - Only runs calibration if no calibration files are found or if overwrite_calibration is True.
    - Uses charis.buildcalibrations to perform the actual calibration.
    """

    if not os.path.exists(wavecal_outputdir):
        os.makedirs(wavecal_outputdir)

    files_in_calibration_folder = glob(os.path.join(wavecal_outputdir, '*key*.fits'))
    if len(files_in_calibration_folder) == 0 or overwrite_calibration:
        calibration_wavelength = instrument.calibration_wavelength
        wavecal_file = observation.frames['WAVECAL']['FILE'][0]
        inImage, hdr = charis.buildcalibrations.read_in_file(
            wavecal_file, instrument, calibration_wavelength,
            ncpus=calibration_parameters['ncpus'])
        charis.buildcalibrations.buildcalibrations(
            inImage=inImage, instrument=instrument,
            inLam=calibration_wavelength.value,
            outdir=wavecal_outputdir,
            header=hdr,
            **calibration_parameters)

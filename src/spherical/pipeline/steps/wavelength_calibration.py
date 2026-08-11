import os
from glob import glob

import charis

from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def run_wavelength_calibration(
    observation,
    instrument,
    calibration_parameters,
    wavecal_outputdir,
    logger,
    overwrite_calibration=False
):
    """Run wavelength calibration for SPHERE/IFS data.

    This is the second step in the SPHERE/IFS data reduction pipeline. It processes
    the wavelength calibration frames to create wavelength calibration products
    needed for subsequent pipeline steps.

    Required Input Files
    -------------------
    From previous step (download_data):
    - IFS/calibration/obs_band/WAVECAL/*.fits
        Wavelength calibration frames from ESO archive

    Generated Output Files
    ---------------------
    In wavecal_outputdir:
    - *key*.fits
        Wavelength calibration key files used by subsequent pipeline steps
        for wavelength calibration of science data

    Parameters
    ----------
    observation : object
        Observation object containing frame information. Must have a 'frames'
        attribute with a 'WAVECAL' key containing the wavelength calibration
        frame information.
    instrument : object
        Instrument object (e.g., charis.instruments.SPHERE). Must have a
        'calibration_wavelength' attribute specifying the calibration wavelength.
    calibration_parameters : dict
        Dictionary of calibration parameters. Must include:
        - 'ncpus': int
            Number of CPUs to use for parallel processing
    wavecal_outputdir : str
        Output directory for wavelength calibration products. Will be created
        if it does not exist.
    overwrite_calibration : bool, optional
        If True, overwrite existing calibration products. Default is False.

    Returns
    -------
    None
        This function performs calibration as a side effect and does not return
        a value.

    Notes
    -----
    - Creates output directories if they do not exist
    - Only runs calibration if no calibration files are found or if
      overwrite_calibration is True
    - Uses charis.buildcalibrations to perform the actual calibration
    - Calibration products are required for wavelength calibration of science
      data in subsequent pipeline steps

    Examples
    --------
    >>> run_wavelength_calibration(
    ...     observation=obs,
    ...     instrument=charis.instruments.SPHERE('YJ'),
    ...     calibration_parameters={'ncpus': 4},
    ...     wavecal_outputdir='/path/to/calibration',
    ...     overwrite_calibration=False
    ... )
    """

    logger.info("Starting wavelength calibration step.", extra={"step": "wavelength_calibration", "status": "started"})
    logger.debug(f"Parameters: instrument={instrument}, calibration_parameters={calibration_parameters}, outputdir={wavecal_outputdir}, overwrite={overwrite_calibration}")

    if not os.path.exists(wavecal_outputdir):
        os.makedirs(wavecal_outputdir)
        logger.debug(f"Created output directory: {wavecal_outputdir}")

    files_in_calibration_folder = glob(os.path.join(wavecal_outputdir, '*key*.fits'))
    if len(files_in_calibration_folder) == 0 or overwrite_calibration:
        try:
            calibration_wavelength = instrument.calibration_wavelength
            wavecal_file = observation.frames['WAVECAL']['FILE'][0]
            logger.debug(f"Using wavecal file: {wavecal_file}, calibration_wavelength: {calibration_wavelength}")
            inImage, hdr = charis.buildcalibrations.read_in_file(
                wavecal_file, instrument, calibration_wavelength,
                ncpus=calibration_parameters['ncpus'])
            charis.buildcalibrations.buildcalibrations(
                inImage=inImage, instrument=instrument,
                inLam=calibration_wavelength.value,
                outdir=wavecal_outputdir,
                header=hdr,
                **calibration_parameters)
            logger.info(f"Wavelength calibration completed and written to: {wavecal_outputdir}")
        except Exception:
            logger.exception("Failed to perform wavelength calibration.", extra={"step": "wavelength_calibration", "status": "failed"})
            return
    else:
        logger.info("Calibration products already exist and overwrite is False. Skipping calibration.")

    logger.info("Finished wavelength calibration step.", extra={"step": "wavelength_calibration", "status": "success"})

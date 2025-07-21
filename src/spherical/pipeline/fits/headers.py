"""
Module which handles populating fits headers from output files of the SPHERICAL and CHARIS pipeline.
"""

__author__ = "L. Welzel"
__all__ = ["update_cube_fits_header_after_reduction"]

import logging
import warnings
from copy import deepcopy
from functools import partial
from itertools import repeat
from os import getlogin
from pathlib import Path
from socket import getfqdn, gethostname
from typing import Callable, Literal

import pandas as pd
from astropy.io.fits.card import VerifyWarning
from astropy.io.fits.hdu.hdulist import fitsopen
from astropy.io.fits.header import Header

LOGGER = logging.getLogger(__name__)


_OUTPUT_GROUP_PATTERNS = [
    "coro",
    "center",
    "flux",
]


def update_cube_fits_header_after_reduction(
    path: Path | str,
    target: Literal["coro", "center", "flux", "all"] = 'all',
    override_mode_file: Literal["copy", "update"] = "update",
    override_mode_header: Literal["keep", "update"] = "update",
) -> None:
    """
    Update the FITS header of reduced data cubes.

    Parameters
    ----------
    path : Path or str
        Directory containing the target *_cube.fits and associated files.
    target : {"coro", "center", "flux", "all"}, default 'all'
        Which reduction product to update; 'all' processes all targets.
    override_mode_file : {"copy", "update"}, default 'update'
        How to handle file overrides.
    override_mode_header : {"keep", "update"}, default 'update'
        How to handle header overrides.

    Raises
    ------
    AssertionError
        If path is invalid or target files are missing.
    NotImplementedError
        If unsupported modes or multiple HDUs are encountered.
    """
    # validate path
    path = Path(path)
    assert path.exists(), f"Path {path} does not exist."
    assert path.is_dir(), f"Path {path} is not a directory."

    # validate target
    assert target in _OUTPUT_GROUP_PATTERNS + ["all"], f"Target {target} is not valid. Must be one of {_OUTPUT_GROUP_PATTERNS} or 'all'."

    # validate override modes
    assert override_mode_file in ["copy", "update"], f"Override mode for file {override_mode_file} is not valid. Must be 'copy' or 'update'."
    assert override_mode_header in ["keep", "update"], f"Override mode for header {override_mode_header} is not valid. Must be 'keep' or 'update'."

    # TODO: implement below
    if override_mode_file == "copy":
        raise NotImplementedError(
            "The 'copy' mode for 'override_mode_file' is not implemented yet. "
            "Please use 'update' mode instead."
        )
    
    # warn if function might lead to data loss
    # TODO: unsure if we should warn about this since it would be the default behavior
    # if override_mode_header == "update" and override_mode_file == "update":
    #     warnings.warn(
    #         "The 'update' mode for 'override_mode_header' together with 'update' mode for 'override_mode_file'"
    #         " is not recommended. This may lead to loss of data in the header.",
    #         UserWarning
    #     )
    
    # dispatch to the function with specific targets if target is "all"
    if target == "all":
        targets = _OUTPUT_GROUP_PATTERNS
        LOGGER.debug(f"Updating headers for all targets: {targets}.")
        for resolved_target in targets:
            LOGGER.debug(f"Updating header for target: {resolved_target}.")
            try:
                update_cube_fits_header_after_reduction(
                    path=path,
                    target=resolved_target,
                    override_mode_file=override_mode_file,
                    override_mode_header=override_mode_header
                )
            except Exception as e:
                LOGGER.error(f"Failed to process header for target {resolved_target}: {e}")
                pass
        return

    # validate target
    target_observation_cube_file_path = path / f"{target}_cube.fits"
    assert target_observation_cube_file_path.exists(), (
        f"Target {target} observation cube file {target_observation_cube_file_path} does not exist. "
        "Please check the path and target."
    )
    assert target_observation_cube_file_path.is_file(), (
        f"Target {target} observation cube file {target_observation_cube_file_path} is not a file. "
        "Please check the path and target."
    )

    # get the fits file
    target_observation_cube_file_path_str = str(target_observation_cube_file_path.resolve().absolute())

    # TODO: unsure if we ever get more than one HDU, but if we do, we should handle it
    try:
        with fitsopen(
            target_observation_cube_file_path_str, 
            ignore_missing_end=False,
            memmap=True,
        ) as hdulist:
            if len(hdulist) != 1:
                hdulist.info()
                raise NotImplementedError("No support for multiple HDUs yet.")
            memmap: bool = True
            fits_header: Header = deepcopy(hdulist[0].header)
            new_header: Header = deepcopy(fits_header)

    except ValueError:
        # If BZERO/BSCALE/BLANK header keywords present HDU can’t load as memmap
        with fitsopen(
            target_observation_cube_file_path_str, 
            ignore_missing_end=False,
            memmap=False,
        ) as hdulist:
            if len(hdulist) != 1:
                hdulist.info()
                raise NotImplementedError("No support for multiple HDUs yet.")
            memmap: bool = False
            fits_header: Header = deepcopy(hdulist[0].header)
            new_header: Header = deepcopy(fits_header)

    try:
        hdulist.close()
        del hdulist[0].data  # free memory
        del hdulist
    except Exception:
        LOGGER.warning("Failed to close HDUList or free memory. This is not critical, but may lead to increase memory usage until the gc runs.")


    # update the header with spherical metadata
    new_header = spherical_populate_fits_header(
        header=new_header, 
        overwrite=override_mode_header == "update",
        include_dependency_metadata=True,
        spherical_key_postfix="POST_PIPE",
    )

    # collect other data files
    frame_info_file_path = path / f"frames_info_{target}.csv"
    frame_info_df = pd.read_csv(frame_info_file_path, index_col=0)

    # get constant columns
    constant_frame_info_csv_columns = find_constant_columns(
        frame_info_df, 
        include_na=False
    )
    # expand the keys to be HIERARCH by default
    constant_frame_info_csv_columns = {
        f"HIERARCH {key}": value for key, value in constant_frame_info_csv_columns.items()
    }

    key_path_to_constant_frame_info_csv_file = "HIERARCH SPHERICAL FRAME_INFO_FILE"
    constant_frame_info_csv_comment = f"constant value from frames_info file ({key_path_to_constant_frame_info_csv_file})"
    extend_fits_header_with_card(
        new_header,
        key=key_path_to_constant_frame_info_csv_file,
        value=str(frame_info_file_path.resolve().absolute()),
        comment="Path to the frames_info file for this target observation.",
        update=override_mode_header == "update",
    )


    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        new_header.extend(
            (
                (k, v, c) for (k, v), c in zip(
                    constant_frame_info_csv_columns.items(), 
                    repeat(constant_frame_info_csv_comment)
                )
            ),
            strip=True,
            update=override_mode_header == "update",
        )

    # update the fits header
    with fitsopen(
        target_observation_cube_file_path_str, 
        ignore_missing_end=False,
        memmap=memmap,
        mode='update',
        output_verify='fix+warn',
    ) as hdulist:
        if len(hdulist) != 1:
            hdulist.info()
            raise NotImplementedError("No support for multiple HDUs yet.")
        hdulist[0].header = new_header

        hdulist.flush(
            output_verify='fix+warn',
        )

    return


def extend_fits_header_with_card(
    header: Header, 
    key: str, 
    value: str | int | float, 
    comment: str | None = None,
    update: bool = True,
    key_prefix: str = "",
) -> Header:
    """
    Add or update a card in a FITS header.

    Parameters
    ----------
    header : Header
        The FITS header to modify.
    key : str
        The keyword to add (HIERARCH prefix applied).
    value : str, int, or float
        The value for the keyword.
    comment : str or None, default None
        Comment for the card.
    update : bool, default True
        Whether to overwrite existing keys.
    key_prefix : str, default ''
        Additional prefix for the keyword.

    Returns
    -------
    Header
        The modified FITS header.
    """

    # validate key
    key = f"{key_prefix} {key}" if key_prefix else key
    key = key.upper()

    # handle HIERARCH explictly to supress warnings
    # if len(key) > 8:
    #     key = f"HIERARCH {key}"

    # handle HIERARCH explictly to supress warnings, and always add the HIERARCH prefix to the key
    key = f"HIERARCH {key}"

    # validate value
    if isinstance(value, str):
        value = value.strip()
        # value = ascii(value)  # convert to ASCII string for FITS compatibility

    # TODO: work around for astropy fits header card bug
    # TODO: tracked in issue: https://github.com/astropy/astropy/issues/17783
    
    if comment is None:
        comment_length = 0
    else:
        comment_length = len(comment)

    if (len(key) + len(str(value)) <= 80) and (len(key) + len(str(value)) + comment_length > 80):
        # key and value fit, but comment does not and will be truncated
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', VerifyWarning)
            header.extend(
                [
                    (key, value, comment),
                    (f"{key} COMMENT", comment),
                ],
                update=update,
            )
        return

    header.extend(
        [(key, value, comment)] if comment else [(key, value)],
        update=update,
    )
    
    return


def spherical_populate_fits_header(
        header: Header | None, 
        overwrite: bool = False,
        include_dependency_metadata: bool = True,
        spherical_key_postfix: str = ""
    ) -> Header:
    """
    Populate a FITS header with SPHERE pipeline metadata.

    Adds general metadata about the SPHERE reduction software,
    pipeline step information, and optional dependency versions.

    Parameters
    ----------
    header : Header or None
        Existing header to update, or None to create a new one.
    overwrite : bool, default False
        If True, overwrite existing spherical keys.
    include_dependency_metadata : bool, default True
        Include versions of Python, charis, esorex, etc.
    spherical_key_postfix : str, default ''
        Suffix appended to spherical metadata keys.

    Returns
    -------
    Header
        The updated FITS header.
    """

    import datetime
    import subprocess
    from importlib.metadata import version

    if header is None:
        header = Header()

    key_prefix = f"SPHERICAL {spherical_key_postfix}" if spherical_key_postfix else "SPHERICAL"

    now = datetime.datetime.now(datetime.timezone.utc).astimezone()

    # bind 'header', 'update' and 'key_prefix="SPHERICAL *"' into a helper 'ac'
    ac_general: Callable[..., Header] = partial(
        extend_fits_header_with_card,
        header,
        update=True,       # does not really matter since we except this part to not change
        key_prefix="SPHERICAL",
    )

    ac: Callable[..., Header] = partial(
        extend_fits_header_with_card,
        header,
        update=not overwrite, # flip bool (overwrite == not update)
        key_prefix=key_prefix,
    )

    # SPHERICAL METADATA GENERAL
    ac_general('DESC', 'VLT/SPHERE Observation Database and IFS Data Analysis Pipeline')
    ac_general('AUTHOR NAME', 'M. Samland', 'author of the package')
    ac_general('AUTHOR EMAIL', 'NA')

    ac_general('PUB NAME', 'Astronomy & Astrophysics, Volume 668, id.A84, 16 pp.')
    ac_general('PUB TITLE', 'Spectral cube extraction for the VLT/SPHERE IFS. Open-source pipeline with full forward modeling and improved sensitivity.')
    ac_general('PUB AUTHORS', 'M. Samland, T. D. Brandt, J. Milli, P. Delorme, and A. Vigan')
    ac_general('PUB YEAR', '2022')
    ac_general('PUB DOI', 'https://doi.org/10.1051/0004-6361/202244587')
    ac_general('PUB ADS', 'https://ui.adsabs.harvard.edu/abs/2022A%26A...668A..84S/abstract')
    
    # SPHERICAL METADATA PIPELINE STEP SPECIFIC
    ac('VERSION', version("spherical"), 'version of the reducer software')
    ac('GIT URL', subprocess.getoutput("git config --get remote.origin.url").strip(), 'url of the git repository')
    ac('GIT HASH', subprocess.getoutput("git rev-parse --short HEAD").strip(), 'git hash')
    ac('GIT BRANCH', subprocess.getoutput("git rev-parse --abbrev-ref HEAD").strip(), 'active git branch')

    # FITS WRITING METADATA
    ac('FITS AUTHOR', getlogin(), 'author of the FITS file')
    ac('FITS HOSTNAME', gethostname(), 'hostname of the machine where the FITS file was written')
    ac('FITS FQDN', getfqdn(), 'fully qualified domain name of the machine where the FITS file was written')
    ac('FITS WRITE DATE', now.strftime("%Y-%m-%d"), 'date of the FITS file writing')
    ac('FITS WRITE TIME',  now.strftime("%H:%M:%S"), 'time of the FITS file writing')


    # DEPENDENCIES
    if include_dependency_metadata:
        import re

        # PYTHON METADATA
        ac('PYTHON VERSION', subprocess.getoutput("python --version").strip())

        # PIPELINE DEPENDENCIES METADATA
        ac('CHARIS VERSION', version("charis"))

        def resolve_esorex_meta(esorex_meta: str) -> dict[str: str]:
            # Extract the version using regex
            version_match = re.search(r"ESO Recipe Execution Tool, version ([\d\.]+)", esorex_meta)
            version = version_match.group(1) if version_match else "UNKNOWN"

            # Extract the libraries line
            libraries_match = re.search(r"Libraries used: (.+)", esorex_meta)
            libraries_line = libraries_match.group(1) if libraries_match else "UNKNOWN"

            # Split and parse the libraries into a dictionary
            libraries = {}
            for lib in libraries_line.split(', '):
                lib_parts = lib.split(' = ')
                if len(lib_parts) == 2:
                    name, ver = lib_parts
                    name = name.strip()
                    ver = ver.strip()
                    libraries[name] = ver

            esorex_meta_dict = {
                "VERSION": version,
                **{
                    f"LIB {name.upper()}": ver for name, ver in libraries.items()
                }
            }

            return esorex_meta_dict
            
        esorex_meta = subprocess.getoutput("esorex --version").strip()
        esorex_meta = resolve_esorex_meta(esorex_meta)
        for key, value in esorex_meta.items():
            ac(f'ESOREX {key}', value)

        

    return header


def find_constant_columns(df: pd.DataFrame, include_na: bool = False) -> dict:
    """
    Identify columns in a DataFrame with constant values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to inspect.
    include_na : bool, default False
        If True, treat NaN as a valid constant.

    Returns
    -------
    dict
        Mapping of column names to their constant value.
    """

    df = df[sorted(df.columns)]

    constants = {}
    for col in df.columns:
        # count unique values, optionally including NaN
        nuniq = df[col].nunique(dropna=not include_na)
        if nuniq == 1:
            # grab that one value (iloc[0] works even if it's NaN)
            constants[col] = df[col].iloc[0]
    return constants


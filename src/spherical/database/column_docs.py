"""Descriptions and units of the columns in the database tables.

The source of truth for the Database columns page of the documentation. A column name
means the same in every table that carries it, except for the few entries in
``OVERRIDES``. Which columns each published table carries is recorded in
``docs/_data/columns/``, written by ``docs/tools/write_column_manifests.py`` at each
table release; ``tests/database/test_column_docs.py`` fails when a published column has
no description here, or a description names no published column.

Standard library only, so the database half keeps working on a base install.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUPS: tuple[str, ...] = (
    "Target identity",
    "Target properties",
    "Gaia enrichment",
    "MOCA enrichment",
    "Instrument setup",
    "Timing",
    "Exposures",
    "Quality flags",
    "Observing conditions",
    "Rotation",
    "Programme",
    "File",
)

TABLE_TYPES: tuple[str, ...] = ("observations", "targets", "other_observations", "files")


@dataclass(frozen=True)
class Column:
    group: str
    description: str
    unit: str = ""


_ID, _PROP, _GAIA, _MOCA, _SETUP, _TIME, _EXP, _FLAG, _COND, _ROT, _PROG, _FILE = GROUPS

COLUMNS: dict[str, Column] = {
    # Target identity (SIMBAD match, target_table.py)
    "MAIN_ID": Column(_ID, "Main SIMBAD identifier of the matched star."),
    "OBJ_HEADER": Column(_ID, "`OBJECT` keyword of the files the star was looked up from."),
    "RA_HEADER": Column(_ID, "Right ascension of the pointing from the file headers.", "deg"),
    "DEC_HEADER": Column(_ID, "Declination of the pointing from the file headers.", "deg"),
    "RA_DEG": Column(_ID, "Right ascension of the star in SIMBAD (ICRS, J2000).", "deg"),
    "DEC_DEG": Column(_ID, "Declination of the star in SIMBAD (ICRS, J2000).", "deg"),
    "ID_HD": Column(_ID, "Henry Draper catalogue identifier, if SIMBAD lists one."),
    "ID_HIP": Column(_ID, "Hipparcos identifier, if SIMBAD lists one."),
    "ID_TYC": Column(_ID, "Tycho identifier, if SIMBAD lists one."),
    "ID_GAIA_DR3": Column(_ID, "Gaia DR3 identifier, if SIMBAD lists one; the key for the Gaia and MOCA enrichment."),
    "ID_2MASS": Column(_ID, "2MASS identifier, if SIMBAD lists one."),
    "POS_DIFF": Column(
        _ID, "Separation between the pointing and the SIMBAD position moved by its proper motion to the epoch of "
        "observation.", "arcsec"
    ),
    "POS_DIFF_ORIG": Column(_ID, "Separation between the pointing and the SIMBAD position, without proper motion.", "arcsec"),
    "STARS_IN_CONE": Column(_ID, "Number of SIMBAD stars within the search cone around the pointing."),
    "FIELD_TARGETS": Column(
        _ID, "Other catalogued targets whose files this sequence took, separated by `|`; those targets have no row "
        "of their own for the sequence."
    ),
    "MATCHED_MAIN_ID": Column(_ID, "Main SIMBAD identifier of the star the set-aside sequence had been matched to."),
    "OBJECT": Column(_ID, "Target name in the `OBJECT` header keyword."),
    "RA": Column(_ID, "Right ascension of the pointing (header `RA`).", "deg"),
    "DEC": Column(_ID, "Declination of the pointing (header `DEC`).", "deg"),
    "TARG_ALPHA": Column(_ID, "Target right ascension given to the telescope, as `HHMMSS.s` (`HIERARCH ESO TEL TARG ALPHA`)."),
    "TARG_DELTA": Column(_ID, "Target declination given to the telescope, as `DDMMSS.s` (`HIERARCH ESO TEL TARG DELTA`)."),
    "TARG_PMA": Column(_ID, "Target proper motion in right ascension given to the telescope (`HIERARCH ESO TEL TARG PMA`)."),
    "TARG_PMD": Column(_ID, "Target proper motion in declination given to the telescope (`HIERARCH ESO TEL TARG PMD`)."),
    # Target properties (SIMBAD)
    "SP_TYPE": Column(_PROP, "Spectral type in SIMBAD."),
    "OTYPE": Column(_PROP, "SIMBAD object type, for example `*`, `**` or `BY*`."),
    "DISTANCE": Column(_PROP, "Distance from the SIMBAD parallax, 1000 / `PLX`.", "pc"),
    "PLX": Column(_PROP, "Parallax in SIMBAD.", "mas"),
    "PLX_ERROR": Column(_PROP, "Uncertainty of the parallax.", "mas"),
    "PLX_BIBCODE": Column(_PROP, "Bibcode of the reference for the parallax."),
    "PMRA": Column(_PROP, "Proper motion in right ascension, including the cos(Dec) factor.", "mas/yr"),
    "PMDEC": Column(_PROP, "Proper motion in declination.", "mas/yr"),
    "PM_ERR_MAJA": Column(_PROP, "Major axis of the proper motion error ellipse.", "mas/yr"),
    "PM_ERR_MINA": Column(_PROP, "Minor axis of the proper motion error ellipse.", "mas/yr"),
    "RV_VALUE": Column(_PROP, "Radial velocity in SIMBAD.", "km/s"),
    "RVZ_ERROR": Column(_PROP, "Uncertainty of the radial velocity.", "km/s"),
    "FLUX_V": Column(_PROP, "V-band magnitude in SIMBAD.", "mag"),
    "FLUX_R": Column(_PROP, "R-band magnitude in SIMBAD.", "mag"),
    "FLUX_I": Column(_PROP, "I-band magnitude in SIMBAD.", "mag"),
    "FLUX_J": Column(_PROP, "J-band magnitude in SIMBAD.", "mag"),
    "FLUX_H": Column(_PROP, "H-band magnitude in SIMBAD.", "mag"),
    "FLUX_K": Column(_PROP, "K-band magnitude in SIMBAD.", "mag"),
    # Gaia DR3 GSP-Phot (gaia_astrophysical_params.py)
    "GAIA_TEFF": Column(_GAIA, "Effective temperature from Gaia DR3 GSP-Phot (`teff_gspphot`).", "K"),
    "GAIA_TEFF_LOWER": Column(_GAIA, "Lower bound of `GAIA_TEFF` (`teff_gspphot_lower`).", "K"),
    "GAIA_TEFF_UPPER": Column(_GAIA, "Upper bound of `GAIA_TEFF` (`teff_gspphot_upper`).", "K"),
    "GAIA_LOGG": Column(_GAIA, "Surface gravity log g from Gaia DR3 GSP-Phot (`logg_gspphot`), g in cm/s².", "dex"),
    "GAIA_LOGG_LOWER": Column(_GAIA, "Lower bound of `GAIA_LOGG` (`logg_gspphot_lower`).", "dex"),
    "GAIA_LOGG_UPPER": Column(_GAIA, "Upper bound of `GAIA_LOGG` (`logg_gspphot_upper`).", "dex"),
    "GAIA_MH": Column(_GAIA, "Metallicity [M/H] from Gaia DR3 GSP-Phot (`mh_gspphot`).", "dex"),
    "GAIA_MH_LOWER": Column(_GAIA, "Lower bound of `GAIA_MH` (`mh_gspphot_lower`).", "dex"),
    "GAIA_MH_UPPER": Column(_GAIA, "Upper bound of `GAIA_MH` (`mh_gspphot_upper`).", "dex"),
    "GAIA_AG": Column(_GAIA, "Extinction in the Gaia G band from GSP-Phot (`ag_gspphot`).", "mag"),
    "GAIA_AG_LOWER": Column(_GAIA, "Lower bound of `GAIA_AG` (`ag_gspphot_lower`).", "mag"),
    "GAIA_AG_UPPER": Column(_GAIA, "Upper bound of `GAIA_AG` (`ag_gspphot_upper`).", "mag"),
    # MOCAdb (mocadb_matching.py); empty when the star has no MOCAdb entry
    "MOCA_ASSOCIATION_NAME": Column(_MOCA, "Name of the young association or cluster the star is assigned to."),
    "MOCA_AGE_MYR": Column(_MOCA, "Adopted age of that association.", "Myr"),
    "MOCA_AGE_MYR_UNC": Column(_MOCA, "Symmetric uncertainty of the association age.", "Myr"),
    "MOCA_AGE_MYR_UNC_POS": Column(_MOCA, "Upper uncertainty of the association age.", "Myr"),
    "MOCA_AGE_MYR_UNC_NEG": Column(_MOCA, "Lower uncertainty of the association age.", "Myr"),
    "MOCA_MEMBERSHIP_TYPE": Column(_MOCA, "MOCAdb membership type code of the star in that association (`moca_mtid`)."),
    "MOCA_BANYAN_PROB": Column(_MOCA, "BANYAN Σ membership probability for the assigned association.", "%"),
    "MOCA_YA_PROB": Column(_MOCA, "MOCAdb probability that the star belongs to any young association (`ya_prob`).", "%"),
    "MOCA_ASSOCIATION_TYPE": Column(_MOCA, "Physical nature of the association, for example `moving group` or `open cluster`."),
    "MOCA_OID": Column(_MOCA, "MOCAdb object identifier of the star."),
    "MOCA_AID": Column(_MOCA, "MOCAdb identifier of the assigned association."),
    "MOCA_AID_SOURCE": Column(
        _MOCA, "Where the association comes from, `banyan` (BANYAN Σ) or `literature`; BANYAN is preferred when both "
        "exist."
    ),
    "MOCA_DESIGNATION": Column(_MOCA, "Designation of the star in MOCAdb."),
    "MOCA_SPECTRAL_TYPE": Column(_MOCA, "Spectral type in MOCAdb."),
    "MOCA_BANYAN_UVW_SEP": Column(_MOCA, "Distance in UVW space velocity from the centre of the assigned association.", "km/s"),
    "MOCA_SPTN": Column(_MOCA, "Spectral type as a number, in the MOCAdb convention (`sptn`)."),
    "MOCA_PARALLAX_MAS": Column(_MOCA, "Parallax adopted by MOCAdb.", "mas"),
    "MOCA_X_PC": Column(_MOCA, "Galactic X position.", "pc"),
    "MOCA_Y_PC": Column(_MOCA, "Galactic Y position.", "pc"),
    "MOCA_Z_PC": Column(_MOCA, "Galactic Z position.", "pc"),
    "MOCA_U_KMS": Column(_MOCA, "Galactic U space velocity.", "km/s"),
    "MOCA_V_KMS": Column(_MOCA, "Galactic V space velocity.", "km/s"),
    "MOCA_W_KMS": Column(_MOCA, "Galactic W space velocity.", "km/s"),
    "MOCA_PROT_DAYS": Column(_MOCA, "Rotation period.", "d"),
    "MOCA_GAIA_ACT": Column(_MOCA, "Gaia DR3 chromospheric activity index (`gaia_act`)."),
    "MOCA_EWLI": Column(_MOCA, "Equivalent width of the lithium 6708 Å line (`ewli`)."),
    "MOCA_EWHA": Column(_MOCA, "Equivalent width of the Hα line (`ewha`)."),
    "MOCA_DR3_RUWE": Column(_MOCA, "Gaia DR3 renormalised unit weight error (RUWE) of the astrometric solution."),
    # Instrument setup
    "INSTRUMENT": Column(_SETUP, "`ifs` or `irdis`."),
    "POLARIMETRY": Column(_SETUP, "True for the IRDIS polarimetry tables."),
    "SPARSE_APERTURE_MASK": Column(_SETUP, "True for the sparse aperture masking tables."),
    "FILTER": Column(_SETUP, "Observing mode: `IFS_MODE` for IFS (`OBS_YJ`, `OBS_H`), `DB_FILTER` for IRDIS (`DB_K12`, `BB_H`)."),
    "ND_FILTER": Column(_SETUP, "Neutral density filter of the science frames, for example `OPEN` or `ND_1.0`."),
    "ND_FILTER_FLUX": Column(_SETUP, "Neutral density filter of the FLUX frames, from the same file as `DIT_FLUX`."),
    "DEROTATOR_MODE": Column(_SETUP, "`PUPIL` for pupil tracking (the field rotates on the detector) or `FIELD` for field tracking."),
    "PRIMARY_SCIENCE": Column(
        _SETUP, "Frame type the science frames are taken from: `CORO` or `CENTER`, whichever holds more exposure time, "
        "or `FLUX` for a sequence stopped before any CORO or CENTER frame."
    ),
    "WAFFLE_MODE": Column(_SETUP, "True when `PRIMARY_SCIENCE` is `CENTER`, a continuous waffle sequence."),
    "SEQ_ARM": Column(_SETUP, "Instrument arm that took the file (`HIERARCH ESO SEQ ARM`)."),
    "CORO": Column(_SETUP, "Coronagraph combination, for example `N_ALC_YJH_S` (`HIERARCH ESO INS COMB ICOR`)."),
    "DB_FILTER": Column(_SETUP, "IRDIS filter combination, for example `DB_H23` (`HIERARCH ESO INS COMB IFLT`)."),
    "POLA": Column(_SETUP, "Polarimetry combination (`HIERARCH ESO INS COMB POLA`)."),
    "INS4 DROT2 MODE": Column(_SETUP, "Derotator mode, `SKY`, `ELEV`, `STAT` or `OFF` (`HIERARCH ESO INS4 DROT2 MODE`)."),
    "SHUTTER": Column(_SETUP, "Shutter state (`HIERARCH ESO INS4 SHUT ST`), True when open."),
    "INS2 MODE": Column(_SETUP, "IFS mode (`HIERARCH ESO INS2 MODE`)."),
    "IFS_MODE": Column(_SETUP, "IFS mode combination, for example `OBS_YJ` or `OBS_H` (`HIERARCH ESO INS2 COMB IFS`)."),
    "PRISM": Column(_SETUP, "IFS prism, `PRIS_YJ` or `PRIS_YJH` (`HIERARCH ESO INS2 OPTI2 ID`)."),
    "LAMP": Column(_SETUP, "IFS calibration lamp (`HIERARCH ESO INS2 CAL`)."),
    "INS1 MODE": Column(_SETUP, "IRDIS mode, for example `DBI`, `CI` or `DPI` (`HIERARCH ESO INS1 MODE`)."),
    "INS1 FILT NAME": Column(_SETUP, "IRDIS common filter (`HIERARCH ESO INS1 FILT NAME`)."),
    "INS1 OPTI2 NAME": Column(_SETUP, "IRDIS dual filter (`HIERARCH ESO INS1 OPTI2 NAME`)."),
    "PAC_X": Column(_SETUP, "IRDIS `HIERARCH ESO INS1 PAC X` position."),
    "PAC_Y": Column(_SETUP, "IRDIS `HIERARCH ESO INS1 PAC Y` position."),
    "DET_ID": Column(_SETUP, "Detector identifier (`HIERARCH ESO DET ID`)."),
    "READOUT_MODE": Column(_SETUP, "Detector readout mode, for example `Double` or `Nondest` (`HIERARCH ESO DET READ CURNAME`)."),
    "WAFFLE_ORIENT": Column(_SETUP, "Orientation of the waffle pattern, `x` or `+` (`HIERARCH ESO OCS WAFFLE ORIENT`)."),
    "SPATIAL_FILTER": Column(_SETUP, "SAXO spatial filter, for example `SMALL` or `LARGE` (`HIERARCH ESO INS4 OPTI22 NAME`)."),
    "AOS TTLOOP STATE": Column(_SETUP, "State of the adaptive optics tip-tilt loop (`HIERARCH ESO AOS TTLOOP STATE`)."),
    "AOS HOLOOP STATE": Column(_SETUP, "State of the adaptive optics high-order loop (`HIERARCH ESO AOS HOLOOP STATE`)."),
    "AOS IRLOOP STATE": Column(_SETUP, "State of the adaptive optics infrared loop (`HIERARCH ESO AOS IRLOOP STATE`)."),
    "AOS PUPLOOP STATE": Column(_SETUP, "State of the adaptive optics pupil loop (`HIERARCH ESO AOS PUPLOOP STATE`)."),
    "AOS VISWFS MODE": Column(_SETUP, "Gain and frequency of the visible wavefront sensor (`HIERARCH ESO AOS VISWFS MODE`)."),
    # Timing
    "NIGHT_START": Column(_TIME, "Date on which the observing night began: the UTC date, one day earlier before 12:00 UT."),
    "OBS_START": Column(_TIME, "Modified Julian date of the first science file.", "d"),
    "OBS_END": Column(_TIME, "Modified Julian date of the last science file.", "d"),
    "MJD_MEAN": Column(_TIME, "Mean modified Julian date of the science files.", "d"),
    "DATE_OBS": Column(_TIME, "Start of the exposure (header `DATE-OBS`), UTC."),
    "DATE_SHORT": Column(_TIME, "UTC date of `DATE_OBS`, as `YYYY-MM-DD`."),
    "MJD_OBS": Column(_TIME, "Modified Julian date of the start of the exposure (header `MJD-OBS`).", "d"),
    "SEQ_UTC": Column(_TIME, "Start of the detector sequence (`HIERARCH ESO DET SEQ UTC`), UTC."),
    "FRAM_UTC": Column(_TIME, "Time stamp of the detector frame (`HIERARCH ESO DET FRAM UTC`), UTC."),
    # Exposures
    "DIT": Column(_EXP, "Exposure time per frame (DIT) of the science frames, from the middle file of the sequence.", "s"),
    "NDIT": Column(_EXP, "Number of frames per science file, from the middle file of the sequence."),
    "NCUBES": Column(_EXP, "Number of science files."),
    "DIT_FLUX": Column(_EXP, "DIT of the FLUX frames; with mixed DITs, the one with the most integration time.", "s"),
    "NDIT_FLUX": Column(_EXP, "Number of frames per FLUX file, from the same file as `DIT_FLUX`."),
    "NFLUX": Column(_EXP, "Number of FLUX files."),
    "DIT_CENTER": Column(_EXP, "DIT of the CENTER frames; 0 when there are none or they mix DITs.", "s"),
    "NDIT_CENTER": Column(_EXP, "Number of frames per CENTER file; 0 when there are none or they mix DITs."),
    "NCENTER": Column(_EXP, "Number of CENTER files."),
    "DIT_CORO": Column(_EXP, "DIT of the CORO frames; 0 when there are none or they mix DITs.", "s"),
    "NDIT_CORO": Column(_EXP, "Number of frames per CORO file; 0 when there are none or they mix DITs."),
    "TOTAL_EXPTIME_SCI": Column(_EXP, "Total exposure time of the science frames.", "min"),
    "TOTAL_EXPTIME_FLUX": Column(_EXP, "Total exposure time of the FLUX frames.", "min"),
    "TOTAL_EXPTIME_CENTER": Column(_EXP, "Total exposure time of the CENTER frames.", "min"),
    "TOTAL_EXPTIME_CORO": Column(_EXP, "Total exposure time of the CORO frames.", "min"),
    "EXPTIME": Column(_EXP, "Exposure time per frame (DIT) of the file (header `EXPTIME`).", "s"),
    "SEQ1_DIT": Column(_EXP, "Detector integration time (`HIERARCH ESO DET SEQ1 DIT`).", "s"),
    "DIT_DELAY": Column(_EXP, "Delay between integrations (`HIERARCH ESO DET DITDELAY`).", "s"),
    # Quality flags (evaluate_observation_flags, compute_hci_ready, match_vetting)
    "HCI_READY": Column(
        _FLAG, "True when the sequence has CENTER and FLUX frames, one DIT across its CENTER frames and one across its "
        "CORO frames, derotation angles that could be computed, and no `VETTING_FLAG`."
    ),
    "VETTING_FLAG": Column(
        _FLAG, "`ambiguous_owner` when some files could belong to either of two nearby targets, `target_changed` when "
        "the target coordinates change within the sequence, empty otherwise."
    ),
    "FLUX_FLAG": Column(_FLAG, "True when the sequence has no FLUX frames."),
    "FLUX_DIT_FLAG": Column(_FLAG, "True when the FLUX frames mix exposure times."),
    "FLUX_ND_FLAG": Column(_FLAG, "True when the FLUX frames mix neutral density filters."),
    "FLUX_DIT_SPREAD": Column(_FLAG, "Ratio of the longest to the shortest FLUX exposure time."),
    "CENTER_FLAG": Column(_FLAG, "True when the sequence has no CENTER frames."),
    "CENTER_DIT_FLAG": Column(_FLAG, "True when the CENTER frames mix exposure times."),
    "CORO_FLAG": Column(_FLAG, "True when the sequence has no CORO frames."),
    "CORO_DIT_FLAG": Column(_FLAG, "True when the CORO frames mix exposure times."),
    "DEROTATOR_FLAG": Column(_FLAG, "True when the derotation angles could not be computed; always True for polarimetry."),
    "CATEGORY": Column(_FLAG, "Why the sequence was set aside: `solar_system`, `non_stellar` or `unmatched`."),
    "VETTING_REASON": Column(_FLAG, "The measured value behind `CATEGORY`, for example `drift 0.24\"/min`."),
    # Observing conditions
    "MEAN_TAU": Column(_COND, "Mean coherence time τ₀ over the science files.", "s"),
    "STDDEV_TAU": Column(_COND, "Standard deviation of τ₀ over the science files.", "s"),
    "MEAN_FWHM": Column(_COND, "Mean seeing over the science files, the average of the start and end values per file.", "arcsec"),
    "STDDEV_FWHM": Column(_COND, "Standard deviation of the seeing over the science files.", "arcsec"),
    "MEAN_AIRMASS": Column(_COND, "Mean airmass at the start of the science files."),
    "AMBI_TAU": Column(_COND, "Coherence time τ₀ (`HIERARCH ESO TEL AMBI TAU0`).", "s"),
    "AMBI_FWHM_START": Column(_COND, "Seeing at the start of the exposure (`HIERARCH ESO TEL AMBI FWHM START`).", "arcsec"),
    "AMBI_FWHM_END": Column(_COND, "Seeing at the end of the exposure (`HIERARCH ESO TEL AMBI FWHM END`).", "arcsec"),
    "AIRMASS": Column(_COND, "Airmass at the start of the exposure (`HIERARCH ESO TEL AIRM START`)."),
    "TEL AIRM END": Column(_COND, "Airmass at the end of the exposure (`HIERARCH ESO TEL AIRM END`)."),
    "TEL IA FWHM": Column(_COND, "Image quality measured by the telescope's image analyser (`HIERARCH ESO TEL IA FWHM`).", "arcsec"),
    "TEL AMBI TEMP": Column(_COND, "Ambient temperature (`HIERARCH ESO TEL AMBI TEMP`).", "deg_C"),
    "TEL AMBI WINDSP": Column(_COND, "Wind speed (`HIERARCH ESO TEL AMBI WINDSP`).", "m/s"),
    "TEL AMBI WINDDIR": Column(_COND, "Wind direction (`HIERARCH ESO TEL AMBI WINDDIR`).", "deg"),
    "ALT_BEGIN": Column(_COND, "Telescope altitude at the start of the exposure (`HIERARCH ESO TEL ALT`).", "deg"),
    "TEL AZ": Column(_COND, "Telescope azimuth at the start of the exposure (`HIERARCH ESO TEL AZ`).", "deg"),
    "TEL GEOELEV": Column(_COND, "Elevation of the telescope site (`HIERARCH ESO TEL GEOELEV`).", "m"),
    "GEOLAT": Column(_COND, "Latitude of the telescope site (`HIERARCH ESO TEL GEOLAT`).", "deg"),
    "GEOLON": Column(_COND, "Longitude of the telescope site (`HIERARCH ESO TEL GEOLON`).", "deg"),
    # Rotation
    "ROTATION": Column(_ROT, "Field rotation, the difference between the derotation angles of the first and last science frame.", "deg"),
    "WAFFLE_AMP": Column(_ROT, "Waffle spot amplitude of the last CENTER file; 0 when there are no CENTER frames."),
    "TEL PARANG START": Column(_ROT, "Parallactic angle at the start of the exposure (`HIERARCH ESO TEL PARANG START`).", "deg"),
    "TEL PARANG END": Column(_ROT, "Parallactic angle at the end of the exposure (`HIERARCH ESO TEL PARANG END`).", "deg"),
    "DROT2_RA": Column(_ROT, "Right ascension the derotator tracks, as `HHMMSS.s` (`HIERARCH ESO INS4 DROT2 RA`)."),
    "DROT2_DEC": Column(_ROT, "Declination the derotator tracks, as `DDMMSS.s` (`HIERARCH ESO INS4 DROT2 DEC`)."),
    "DROT2_BEGIN": Column(_ROT, "Derotator position at the start of the exposure (`HIERARCH ESO INS4 DROT2 BEGIN`).", "deg"),
    "DROT2_END": Column(_ROT, "Derotator position at the end of the exposure (`HIERARCH ESO INS4 DROT2 END`).", "deg"),
    "INS4 DROT2 POSANG": Column(_ROT, "Position angle set on the derotator (`HIERARCH ESO INS4 DROT2 POSANG`).", "deg"),
    # Programme
    "OBS_PROG_ID": Column(_PROG, "ESO programme identifier, for example `095.C-0298(A)`, from the middle science file."),
    "OBS_ID": Column(_PROG, "ESO observation block identifier, from the middle science file."),
    "OBS_NAME": Column(_PROG, "Name of the observation block (`HIERARCH ESO OBS NAME`)."),
    "OBS_PI_COI": Column(_PROG, "Name of the PI or co-investigator of the programme (`HIERARCH ESO OBS PI-COI NAME`)."),
    "TOTAL_FILE_SIZE_MB": Column(_FILE, "Total size of the science files, estimated from their headers.", "MB"),
    # File
    "DPR_CATG": Column(_FILE, "Data category, `SCIENCE` or `CALIB` (`HIERARCH ESO DPR CATG`)."),
    "DPR_TYPE": Column(
        _FILE, "Frame type: `OBJECT` for CORO, `OBJECT,CENTER`, `OBJECT,FLUX`, or a calibration type such as "
        "`FLAT,LAMP` (`HIERARCH ESO DPR TYPE`)."
    ),
    "DPR_TECH": Column(_FILE, "Observing technique, for example `IMAGE,DUAL,CORONOGRAPHY` or `IFU` (`HIERARCH ESO DPR TECH`)."),
    "ORIGFILE": Column(_FILE, "Original file name at the telescope (header `ORIGFILE`)."),
    "DP.ID": Column(_FILE, "ESO archive file identifier, used to download the file."),
    "DATE": Column(_FILE, "Date the file was written (header `DATE`), UTC."),
    "FILE_SIZE": Column(_FILE, "Size of the file, estimated from its header.", "MB"),
    "NAXIS": Column(_FILE, "Number of data axes (header `NAXIS`)."),
    "NAXIS1": Column(_FILE, "Length of the first data axis (header `NAXIS1`).", "pix"),
    "NAXIS2": Column(_FILE, "Length of the second data axis (header `NAXIS2`).", "pix"),
    "NAXIS3": Column(_FILE, "Number of frames in the file (header `NAXIS3`)."),
    "BITPIX": Column(_FILE, "Bits per data value (header `BITPIX`)."),
}

# Columns whose meaning in one table differs from the entry in COLUMNS.
OVERRIDES: dict[tuple[str, str], Column] = {
    ("files", "NDIT"): Column(_EXP, "Number of integrations requested from the detector (`HIERARCH ESO DET NDIT`)."),
    ("files", "ND_FILTER"): Column(_SETUP, "Neutral density filter (`HIERARCH ESO INS4 FILT2 NAME`)."),
    ("files", "DEROTATOR_MODE"): Column(_SETUP, "Derotator tracking mode, `PUPIL` or `FIELD` (`HIERARCH ESO INS4 COMB ROT`)."),
    ("files", "WAFFLE_AMP"): Column(_SETUP, "Waffle spot amplitude (`HIERARCH ESO OCS WAFFLE AMPL`)."),
    ("files", "OBS_PROG_ID"): Column(_PROG, "ESO programme identifier (`HIERARCH ESO OBS PROG ID`)."),
    ("files", "OBS_ID"): Column(_PROG, "ESO observation block identifier (`HIERARCH ESO OBS ID`)."),
    ("other_observations", "OBJECT"): Column(_ID, "The `OBJECT` header names of the sequence, separated by `|`."),
}


def describe(table_type: str, name: str) -> Column:
    """The description of column ``name`` in tables of type ``table_type``; ``KeyError`` if there is none."""
    return OVERRIDES.get((table_type, name)) or COLUMNS[name]

"""Assign science files to targets and set aside sequences that are not stellar HCI observations (#224).

The SIMBAD match decides which star each target is. This module decides which target
owns each science file, and recognises sequences whose match is wrong: moving
Solar-system targets, non-stellar objects, and matches far from the pointing whose header
name does not fit. ``observation_table.create_observation_table`` calls it.
"""

import csv
import re
from dataclasses import dataclass
from importlib.resources import files as package_files
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from spherical.database.database_utils import SIMBAD_TYPE_PREFIX, normalize_name

#: Median header drift (arcsec/min) above which a steadily moving sequence is a Solar-system target.
#: In the v3 tables every mover found moves at >= 0.049"/min, and the fastest sidereal pointing
#: creep is 0.019"/min (TW Hya, three SAM frames over an hour).
DRIFT_RATE_MIN = 0.03
#: SIMBAD match offset (arcsec) above which a sequence without a name match is ``unmatched``.
MAX_MATCH_OFFSET = 12.0
#: Two candidate owners closer than this (arcsec) in distance to the pointing are ambiguous.
AMBIGUOUS_OWNER_SEP = 1.0
#: Spread (arcsec) of the OB target coordinates within a sequence that flags ``target_changed``.
TARGET_CHANGE_SEP = 2.0
#: OB coordinates of one star can differ by its proper motion over this many years, when two OBs
#: give them at different epochs (Sirius: 9.6" apart on one night).
TARGET_CHANGE_EPOCH_YEARS = 30.0
#: Names listed in ``FIELD_TARGETS`` before the rest is summarised as ``+N more``.
MAX_FIELD_TARGETS = 5

ID_COLUMNS = ("MAIN_ID", "ID_HD", "ID_HIP", "ID_TYC", "ID_2MASS", "ID_GAIA_DR3")
CATEGORIES = ("solar_system", "non_stellar", "unmatched")
_EMPTY_NAMES = {"", "nan", "none", "--", "n/a"}


def text(value) -> str:
    """A table cell as a stripped string; masked cells and bytes are handled."""
    if np.ma.is_masked(value):
        return ""
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip()


def as_float(value) -> float:
    """A table cell as a float, NaN when masked or not numeric."""
    if np.ma.is_masked(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_designation(name) -> str:
    """Compare-ready form of a target name: SIMBAD type prefix dropped, HDE read as HD."""
    designation = text(name)
    if designation.lower() in _EMPTY_NAMES:
        return ""
    designation = SIMBAD_TYPE_PREFIX.sub("", designation)
    designation = re.sub(r"(?i)^HDE(?=[\s_]*\d)", "HD", designation)
    return normalize_name(designation)


def target_identifiers(target) -> set[str]:
    """Normalised designations of a target row, from ``ID_COLUMNS`` split on ``|``."""
    identifiers = set()
    for column in ID_COLUMNS:
        if column not in target.colnames:
            continue
        for designation in text(target[column]).split("|"):
            normalised = normalize_designation(designation)
            if normalised:
                identifiers.add(normalised)
    return identifiers


# Frame-to-frame steps larger than this (arcsec) are re-pointings, not motion.
_REPOINT_STEP = 30.0
# Steps smaller than this (arcsec) count as standing still.
_MOVING_STEP = 0.01
# Frames closer in time than this (minutes) give no usable rate.
_MIN_STEP_MINUTES = 0.05
_MIN_FRAMES = 3
_MIN_MOVING_FRACTION = 0.8
_MIN_STRAIGHTNESS = 0.8


def steady_drift_rate(ra_deg, dec_deg, mjd) -> float:
    """Median header drift (arcsec/min) of a pointing that moves steadily, else 0.0.

    A tracked Solar-system target moves the header RA/DEC a little in every frame and
    in one direction. Re-acquisitions, toggles between binary components and alternation
    between two targets under one name move it in jumps or back and forth, which the
    moving-fraction and straightness conditions reject.
    """
    mjd = np.asarray(mjd, dtype=float)
    order = np.argsort(mjd)
    mjd = mjd[order]
    ra = np.asarray(ra_deg, dtype=float)[order]
    dec = np.asarray(dec_deg, dtype=float)[order]
    if len(mjd) < _MIN_FRAMES:
        return 0.0
    x = ra * np.cos(np.deg2rad(np.median(dec))) * 3600.0
    y = dec * 3600.0
    minutes = np.diff(mjd) * 1440.0
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    usable = (minutes > _MIN_STEP_MINUTES) & (step < _REPOINT_STEP)
    if usable.sum() < _MIN_FRAMES - 1:
        return 0.0
    step, minutes, dx, dy = step[usable], minutes[usable], dx[usable], dy[usable]
    if np.mean(step > _MOVING_STEP) < _MIN_MOVING_FRACTION:
        return 0.0
    path = step.sum()
    if np.hypot(dx.sum(), dy.sum()) < _MIN_STRAIGHTNESS * path:
        return 0.0
    return float(np.median(step / minutes))


def moving_sequences(t_science) -> dict[tuple[str, str], float]:
    """Drift rate of every ``(OBJECT, NIGHT_START)`` group of science files that moves steadily."""
    moving: dict[tuple[str, str], float] = {}
    if len(t_science) == 0:
        return moving
    grouped = t_science.group_by(["OBJECT", "NIGHT_START"])
    for key, group in zip(grouped.groups.keys, grouped.groups):
        rate = steady_drift_rate(group["RA"], group["DEC"], group["MJD_OBS"])
        if rate > DRIFT_RATE_MIN:
            moving[(text(key["OBJECT"]), text(key["NIGHT_START"]))] = rate
    return moving


def vetting_parameters() -> dict:
    """The vetting constants, for the provenance ``build_parameters``."""
    return {
        "drift_rate_min": DRIFT_RATE_MIN,
        "max_match_offset": MAX_MATCH_OFFSET,
        "ambiguous_owner_sep": AMBIGUOUS_OWNER_SEP,
        "target_change_sep": TARGET_CHANGE_SEP,
        "target_change_epoch_years": TARGET_CHANGE_EPOCH_YEARS,
        "max_field_targets": MAX_FIELD_TARGETS,
    }


@dataclass(frozen=True)
class Override:
    """A curated decision for sequences whose header ``OBJECT`` matches ``pattern``."""

    pattern: re.Pattern
    category: str
    reason: str


def load_overrides(path=None) -> list[Override]:
    """Read the override CSV (default: the packaged ``data/match_overrides.csv``)."""
    source = package_files("spherical.database") / "data" / "match_overrides.csv" if path is None else Path(path)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.lstrip().startswith("#")))
    overrides = []
    for row in rows:
        category = row["category"].strip()
        if category not in ("solar_system", "non_stellar", "keep"):
            raise ValueError(f"Unknown override category {category!r} for pattern {row['pattern']!r}")
        overrides.append(Override(re.compile(row["pattern"], re.IGNORECASE), category, row["reason"].strip()))
    return overrides


def match_override(object_names, overrides) -> Override | None:
    """The first override whose pattern matches one of the names, in file order."""
    for override in overrides:
        if any(override.pattern.search(name) for name in object_names):
            return override
    return None


def classify_sequence(object_names, night, pos_diff, identifiers, moving, overrides) -> tuple[str, str] | None:
    """``(category, reason)`` for a sequence that is not a stellar HCI observation, else None.

    Precedence: an override decides alone (``keep`` exempts the sequence), then steady
    header drift, then a match offset above ``MAX_MATCH_OFFSET`` without a header name
    that matches the target's identifiers.
    """
    override = match_override(object_names, overrides)
    if override is not None:
        return None if override.category == "keep" else (override.category, f"override: {override.reason}")
    rates = [moving[(name, night)] for name in object_names if (name, night) in moving]
    if rates:
        return "solar_system", f'drift {max(rates):.2f}"/min'
    named = any(normalize_designation(name) in identifiers for name in object_names)
    if np.isfinite(pos_diff) and pos_diff > MAX_MATCH_OFFSET and not named:
        return "unmatched", f'offset {pos_diff:.0f}" no name match'
    return None


def _sexagesimal_deg(values, hours: bool) -> np.ndarray:
    """ESO ``TARG_ALPHA`` (HHMMSS.s) or ``TARG_DELTA`` (DDMMSS.s) numbers in degrees."""
    values = np.asarray(values, dtype=float)
    sign = np.where(values < 0, -1.0, 1.0)
    values = np.abs(values)
    whole = np.floor(values / 1e4)
    minutes = np.floor((values - whole * 1e4) / 100.0)
    seconds = values - whole * 1e4 - minutes * 100.0
    degrees = sign * (whole + minutes / 60.0 + seconds / 3600.0)
    return degrees * 15.0 if hours else degrees


def target_coordinate_spread(files) -> float:
    """Largest distance (arcsec) of a file's OB target coordinates from their median.

    The header RA/DEC wander by several arcsec with re-acquisitions; the OB target
    coordinates change only when the OB points at another star.
    """
    if len(files) < 2 or "TARG_ALPHA" not in files.colnames or "TARG_DELTA" not in files.colnames:
        return 0.0
    alpha = np.ma.filled(np.ma.asarray(files["TARG_ALPHA"], dtype=float), np.nan)
    delta = np.ma.filled(np.ma.asarray(files["TARG_DELTA"], dtype=float), np.nan)
    good = np.isfinite(alpha) & np.isfinite(delta)
    if good.sum() < 2:
        return 0.0
    ra = _sexagesimal_deg(alpha[good], hours=True)
    dec = _sexagesimal_deg(delta[good], hours=False)
    ra0, dec0 = np.median(ra), np.median(dec)
    dx = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600.0
    dy = (dec - dec0) * 3600.0
    return float(np.max(np.hypot(dx, dy)))


def target_change_limit(pmra_mas_yr, pmdec_mas_yr) -> float:
    """Spread (arcsec) of the OB target coordinates above which a sequence mixes two stars."""
    proper_motion = np.hypot(pmra_mas_yr, pmdec_mas_yr) / 1000.0
    if not np.isfinite(proper_motion):
        return TARGET_CHANGE_SEP
    return max(TARGET_CHANGE_SEP, TARGET_CHANGE_EPOCH_YEARS * proper_motion)


def format_field_targets(names) -> str:
    """``|``-joined sorted unique names, at most ``MAX_FIELD_TARGETS`` then ``+N more``."""
    names = sorted({name for name in names if name})
    if len(names) <= MAX_FIELD_TARGETS:
        return "|".join(names)
    return "|".join(names[:MAX_FIELD_TARGETS]) + f"|+{len(names) - MAX_FIELD_TARGETS} more"


_J2000_MJD = 51544.5
_MAS_PER_DEG = 3.6e6


@dataclass
class FileOwnership:
    """Which target owns each science file (``owner``, -1 when none) and who else could have."""

    owner: np.ndarray
    ambiguous: np.ndarray
    others: list
    has_candidate_files: np.ndarray


def _column(table, name, fill) -> np.ndarray:
    """A numeric column as floats, masked and missing values replaced by ``fill``."""
    if name not in table.colnames:
        return np.full(len(table), fill, dtype=float)
    values = np.ma.filled(np.ma.asarray(table[name], dtype=float), np.nan)
    return np.where(np.isfinite(values), values, fill)


def _simbad_separation_arcsec(targets, candidates, ra, dec, mjd) -> np.ndarray:
    """Separation of a pointing from each candidate's SIMBAD position at the file's epoch."""
    head_ra = _column(targets, "RA_HEADER", np.nan)[candidates]
    head_dec = _column(targets, "DEC_HEADER", np.nan)[candidates]
    ra0 = _column(targets, "RA_DEG", np.nan)[candidates]
    dec0 = _column(targets, "DEC_DEG", np.nan)[candidates]
    ra0 = np.where(np.isfinite(ra0), ra0, head_ra)
    dec0 = np.where(np.isfinite(dec0), dec0, head_dec)
    years = (mjd - _J2000_MJD) / 365.25
    pmra = _column(targets, "PMRA", 0.0)[candidates]
    pmdec = _column(targets, "PMDEC", 0.0)[candidates]
    dec_epoch = dec0 + pmdec * years / _MAS_PER_DEG
    ra_epoch = ra0 + pmra * years / _MAS_PER_DEG / np.cos(np.deg2rad(dec0))
    pointing = SkyCoord(ra * u.deg, dec * u.deg)
    return SkyCoord(ra_epoch * u.deg, dec_epoch * u.deg).separation(pointing).arcsec


def _separation_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle distance (arcsec) between positions in degrees (haversine)."""
    ra1, dec1, ra2, dec2 = (np.deg2rad(np.asarray(v, dtype=float)) for v in (ra1, dec1, ra2, dec2))
    h = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))) * 3600.0


def _cone_pairs(file_ra, file_dec, target_ra, target_dec, radius_arcsec) -> tuple[np.ndarray, np.ndarray]:
    """``(file_index, target_index)`` of every file strictly within ``radius_arcsec`` of a target.

    astropy's ``search_around_sky`` needs scipy, which the database install does not have,
    and fails on NaN. Files sorted by declination give each target a declination window
    instead; rows with non-finite coordinates are never matched.
    """
    radius_deg = radius_arcsec / 3600.0
    finite = np.flatnonzero(np.isfinite(file_ra) & np.isfinite(file_dec))
    by_dec = finite[np.argsort(file_dec[finite], kind="stable")]
    sorted_dec = file_dec[by_dec]
    file_parts, target_parts = [], []
    for t in np.flatnonzero(np.isfinite(target_ra) & np.isfinite(target_dec)):
        low = np.searchsorted(sorted_dec, target_dec[t] - radius_deg, side="left")
        high = np.searchsorted(sorted_dec, target_dec[t] + radius_deg, side="right")
        window = by_dec[low:high]
        inside = window[_separation_arcsec(file_ra[window], file_dec[window], target_ra[t], target_dec[t]) < radius_arcsec]
        file_parts.append(inside)
        target_parts.append(np.full(len(inside), t, dtype=int))
    if not file_parts:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(file_parts), np.concatenate(target_parts)


def assign_file_owners(t_science, targets, cone, identifiers) -> FileOwnership:
    """Give each science file to one of the targets whose header cone contains it.

    One candidate owns the file, as before ownership existed. With several, the candidate
    whose identifiers match the file's ``OBJECT`` owns it if exactly one does; otherwise
    the candidate whose SIMBAD position (moved by its proper motion to the file's date) is
    closest to the file's header RA/DEC. When distance decides and the two closest are
    within ``AMBIGUOUS_OWNER_SEP`` of each other, the file is marked ambiguous.
    """
    n_files, n_targets = len(t_science), len(targets)
    owner = np.full(n_files, -1, dtype=int)
    ambiguous = np.zeros(n_files, dtype=bool)
    others: list = [() for _ in range(n_files)]
    has_candidate_files = np.zeros(n_targets, dtype=bool)
    if n_files == 0 or n_targets == 0:
        return FileOwnership(owner, ambiguous, others, has_candidate_files)

    file_ra = _column(t_science, "RA", np.nan)
    file_dec = _column(t_science, "DEC", np.nan)
    file_mjd = _column(t_science, "MJD_OBS", np.nan)
    file_index, target_index = _cone_pairs(
        file_ra, file_dec, _column(targets, "RA_HEADER", np.nan), _column(targets, "DEC_HEADER", np.nan),
        cone.to_value(u.arcsec),
    )
    has_candidate_files[np.unique(target_index)] = True

    order = np.lexsort((target_index, file_index))
    file_index, target_index = file_index[order], target_index[order]
    starts = np.flatnonzero(np.r_[True, np.diff(file_index) != 0]) if len(file_index) else np.array([], dtype=int)
    ends = np.r_[starts[1:], len(file_index)].astype(int)

    single = (ends - starts) == 1
    owner[file_index[starts[single]]] = target_index[starts[single]]
    for start, end in zip(starts[~single], ends[~single]):
        f = file_index[start]
        candidates = target_index[start:end]
        name = normalize_designation(t_science["OBJECT"][f])
        named = [c for c in candidates if name and name in identifiers[c]]
        if len(named) == 1:
            owner[f] = named[0]
        else:
            seps = _simbad_separation_arcsec(targets, candidates, file_ra[f], file_dec[f], file_mjd[f])
            ranked = np.argsort(seps)
            owner[f] = candidates[ranked[0]]
            ambiguous[f] = seps[ranked[1]] - seps[ranked[0]] < AMBIGUOUS_OWNER_SEP
        others[f] = tuple(sorted(int(c) for c in candidates if c != owner[f]))
    return FileOwnership(owner, ambiguous, others, has_candidate_files)

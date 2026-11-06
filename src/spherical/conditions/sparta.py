"""Read SPHERE SPARTA files into time series with explicit sample intervals.

Only numpy, pandas and astropy are used, so the module imports in a base install.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger(__name__)

SPARTA_TABLES: dict[str, tuple[str, ...]] = {
    "AtmPerfParams": ("R0", "T0", "L0", "WindSpeed", "StrehlRatio"),
    "VisLoopParams": ("Flux_avg", "Flux_std"),
    "IRLoopParams": ("Flux_avg", "Flux_std"),
}

CONFIG_KEYS: dict[str, str] = {
    "wfs_mode": "HIERARCH ESO AOS VISWFS MODE",
    "spatial_filter": "HIERARCH ESO INS4 OPTI22 NAME",
    "wfs_filter": "HIERARCH ESO INS4 FILT3 NAME",
    "nd_filter": "HIERARCH ESO INS4 FILT2 NAME",
    "ob_id": "HIERARCH ESO OBS ID",
}

DTTS_EXTENSION = "IRPixelAvgFrame"

# Sample interval used only when a table has a single distinct sample; otherwise the median spacing of
# the distinct samples of that file is used. The fallbacks are the cadences seen on the 51 Eri nights of
# 2015-09-25 and 2017-09-28. The DTTS cadence changed between them (29.75 s in 2015, 2 s in 2017).
DEFAULT_CADENCE_S: dict[str, float] = {
    "AtmPerfParams": 20.0,
    "VisLoopParams": 30.0,
    "IRLoopParams": 30.0,
    "IRPixelAvgFrame": 2.0,
}

# Where a time stamp sits in the averaging interval it labels. VisLoopParams repeats each Flux_avg on
# the next row (stamps 10 s and 20 s apart in turn), so its distinct values arrive every 30 s; the
# repeats are dropped before the intervals are set. Determined on the 51 Eri night of 2017-09-28 (six
# files from 08:18:45 to 09:27:54) by averaging the 2 s DTTS image sums (border-median background
# removed) over the window ending at, centred on and starting at each kept time stamp, and regressing
# Flux_avg on that mean. Relative RMS of the residual over the six files:
#   IRLoopParams (30 s):  end 0.0005-0.0020, mid 0.0024-0.0486, start 0.0041-0.0629; best lag -2 to 0 s
#   VisLoopParams (30 s): end 0.0065-0.0548, mid 0.0050-0.0362, start 0.0074-0.0610; best lag +11 to +20 s
# The end convention wins in every file for IRLoopParams and the mid convention in every file for
# VisLoopParams. On 2015-09-25 (file 09:36:43, DTTS every 29.75 s) the same order holds, and DTTS stamps
# taken as interval ends give the IRLoopParams fit with end stamps its lowest residual (0.0070 against
# 0.0111 for mid and 0.0145 for start), so IRPixelAvgFrame stamps are interval ends. AtmPerfParams has no
# flux; correlating its StrehlRatio with the DTTS peak-to-sum ratio does not separate end from mid
# (correlation 0.45-0.80 against 0.29-0.76; start is lower in every file), so it follows VisLoopParams,
# which comes from the same wavefront sensor.
SAMPLE_STAMP: dict[str, str] = {
    "AtmPerfParams": "mid",
    "VisLoopParams": "mid",
    "IRLoopParams": "end",
    "IRPixelAvgFrame": "end",
}

PHOTONS_PER_ADU = 17.0
_DEFAULT_DTTS_SIDE = 32
# 90% of the DTTS star's flux lies within about 6 px. SPARTA stores averages of frames whose negative
# pixels were set to zero, so every pixel without starlight carries a positive bias of about 0.3 ADU
# (issue #294). The mean of a sigma-clipped ring measures that bias; a median does not.
DTTS_APERTURE_RADIUS = 7.0
DTTS_RING_RADII = (10.0, 14.0)
# Clipped noise is one-sided and often mostly zeros, so only high outliers (stars, ghosts, hot pixels)
# are rejected, around the mean and standard deviation; the MAD is zero when most pixels are zero.
DTTS_RING_CLIP_SIGMA = 5.0
# The tip-tilt loop holds the star close to the window centre; a centre found further out is noise.
_DTTS_MAX_CENTRE_OFFSET = 3


@dataclass
class SpartaFile:
    """Parsed content of one SPARTA file.

    Attributes:
        dp_id: File stem, e.g. ``SPHER.2017-09-28T09:14:03.195``.
        config: Header values from ``CONFIG_KEYS`` (``None`` when absent) plus ``wfs_gain``.
        tables: One DataFrame per entry of ``SPARTA_TABLES`` with ``mjd_start``, ``mjd_end`` and
            the value columns. VisLoopParams also has ``flux_subap``.
        dtts: DTTS photometry per 2 s image with ``mjd_start``, ``mjd_end``, ``dtts_flux`` and
            ``dtts_peak``.
        dtts_images: DTTS images with shape (n, ny, nx), background not subtracted.
        missing: Extension names and ``extension.column`` entries that were not found.
    """

    dp_id: str
    config: dict[str, str | int | float | None]
    tables: dict[str, pd.DataFrame]
    dtts: pd.DataFrame
    dtts_images: np.ndarray
    missing: list[str]


def wfs_gain(mode: str | None) -> float:
    """Return the EM gain from a WFS mode string such as ``GAIN_1000_FREQ_1380Hz``, NaN if unparsable."""
    match = re.search(r"GAIN_(\d+)", mode or "")
    return float(match.group(1)) if match else float("nan")


def _native(column: np.ndarray) -> np.ndarray:
    return column.astype(column.dtype.newbyteorder("="))


def _intervals(name: str, seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MJD start and end of each sample, following ``SAMPLE_STAMP`` and the table's median spacing."""
    default = DEFAULT_CADENCE_S[name]
    step = float(np.median(np.diff(seconds))) if len(seconds) > 1 else default
    if not step > 0:
        step = default
    stamp = SAMPLE_STAMP.get(name, "end")
    offset = {"start": 0.0, "mid": -step / 2, "end": -step}[stamp]
    start = Time(seconds + offset, format="unix").mjd if len(seconds) else np.empty(0)
    return start, start + step / 86400.0


def _seconds(data) -> np.ndarray:
    return _native(data["Sec"]).astype(float) + _native(data["USec"]).astype(float) * 1e-6


def _changed_rows(values: list[np.ndarray]) -> np.ndarray:
    """Rows whose values differ from the previous row; SPARTA repeats a value until the next one is computed."""
    n = len(values[0]) if values else 0
    keep = np.ones(n, dtype=bool)
    if values and n > 1:
        same = np.ones(n - 1, dtype=bool)
        for v in values:
            same &= (v[1:] == v[:-1]) | (np.isnan(v[1:]) & np.isnan(v[:-1]))
        keep[1:] = ~same
    return keep


def _read_table(name: str, hdu, columns: tuple[str, ...], missing: list[str]) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["mjd_start", "mjd_end", *columns], dtype=float)
    data = hdu.data
    if data is None or len(data) == 0:
        names = hdu.columns.names if hdu.columns is not None else []
        missing.extend(f"{name}.{c}" for c in columns if c not in names)
        return empty
    names = hdu.columns.names
    absent_time = [f"{name}.{c}" for c in ("Sec", "USec") if c not in names]
    if absent_time:
        missing.extend(absent_time)
        return empty
    values = {}
    for column in columns:
        if column in names:
            values[column] = _native(data[column]).astype(float)
        else:
            missing.append(f"{name}.{column}")
    keep = _changed_rows(list(values.values())) if values else np.ones(len(data), dtype=bool)
    start, end = _intervals(name, _seconds(data)[keep])
    frame = pd.DataFrame({"mjd_start": start, "mjd_end": end})
    for column in columns:
        frame[column] = values[column][keep] if column in values else np.nan
    return frame


def _dtts_centre(images: np.ndarray) -> tuple[float, float]:
    """Star position from the mean of the images, kept within a few pixels of the window centre.

    A file without the star (taken during FLUX frames) gives a centre on noise, which is harmless because
    the ring background is unbiased wherever the aperture sits.
    """
    ny, nx = images.shape[1:]
    centre = np.array([(ny - 1) / 2, (nx - 1) / 2])
    finite = np.isfinite(images)
    mean = np.where(finite, images, 0.0).sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
    padded = np.pad(mean, 1)
    smooth = sum(padded[dy : dy + ny, dx : dx + nx] for dy in range(3) for dx in range(3))
    peak = np.array(np.unravel_index(np.argmax(smooth), smooth.shape), dtype=float)
    offset = np.clip(peak - centre, -_DTTS_MAX_CENTRE_OFFSET, _DTTS_MAX_CENTRE_OFFSET)
    return tuple(centre + offset)


def _ring_background(values: np.ndarray) -> float:
    """Mean of the ring pixels after iteratively rejecting high outliers."""
    values = values[np.isfinite(values)]
    while values.size > 1:
        keep = values <= values.mean() + DTTS_RING_CLIP_SIGMA * values.std()
        if keep.all():
            break
        values = values[keep]
    return float(values.mean()) if values.size else float("nan")


def dtts_photometry(images: np.ndarray) -> pd.DataFrame:
    """Aperture flux and peak of each DTTS image.

    The flux is the sum within ``DTTS_APERTURE_RADIUS`` of the star, minus the aperture area times the
    sigma-clipped mean of a ring between ``DTTS_RING_RADII``. The star position is found once from the
    mean of all images, since the tip-tilt loop holds it fixed. Aperture pixels that carry starlight
    lose less to SPARTA's clipping than the ring, so the flux reads low by at most the aperture area
    times the ring mean, about 50 ADU: 5% for the faintest stars seen (about 600 ADU), under 1% for
    bright ones.

    The peak is the vertex of a 2D quadratic fitted by least squares to the 3x3 pixels around the
    brightest pixel. When that quadratic is not negative definite (a flat top) or its vertex leaves the
    3x3 window, the fitted value at the brightest pixel is used. NaN pixels are ignored for the
    background and the peak position; a NaN pixel inside the 3x3 window gives the brightest pixel as the
    peak, and a NaN pixel inside the aperture makes the flux NaN. The radii suit the 32x32 window of
    every SPHERE file seen; a window too small to hold the ring gives a NaN flux.

    Args:
        images: Array of shape (n, ny, nx).

    Returns:
        DataFrame with columns ``dtts_flux`` and ``dtts_peak``.
    """
    images = np.asarray(images, dtype=float)
    fluxes = np.full(len(images), np.nan)
    peaks = np.empty(len(images))
    if len(images):
        cy, cx = _dtts_centre(images)
        yy, xx = np.indices(images.shape[1:])
        radius = np.hypot(yy - cy, xx - cx)
        aperture = radius <= DTTS_APERTURE_RADIUS
        ring = (radius >= DTTS_RING_RADII[0]) & (radius <= DTTS_RING_RADII[1])
    yy, xx = np.mgrid[-1:2, -1:2]
    design = np.column_stack([xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel(), xx.ravel(), yy.ravel(), np.ones(9)])
    for i, image in enumerate(images):
        background = _ring_background(image[ring])
        if np.isfinite(background):
            fluxes[i] = image[aperture].sum() - aperture.sum() * background
            image = image - background
        if not np.isfinite(image).any():
            peaks[i] = np.nan
            continue
        # The centre of all pixels tied at the maximum keeps a flat-topped block centred.
        tied = np.argwhere(image == np.nanmax(image))
        row, col = np.clip(np.rint(tied.mean(axis=0)).astype(int), 1, np.array(image.shape) - 2)
        patch = image[row - 1 : row + 2, col - 1 : col + 2].ravel()
        if not np.isfinite(patch).all():
            peaks[i] = np.nanmax(patch)
            continue
        a, b, c, d, e, f = np.linalg.lstsq(design, patch, rcond=None)[0]
        peaks[i] = f
        if a < 0 and 4 * a * b - c * c > 0:
            vertex = np.linalg.solve([[2 * a, c], [c, 2 * b]], [-d, -e])
            if np.all(np.abs(vertex) <= 1.5):
                peaks[i] = f + 0.5 * (d * vertex[0] + e * vertex[1])
    return pd.DataFrame({"dtts_flux": fluxes, "dtts_peak": peaks})


def derive_atmospheric(atm: pd.DataFrame) -> pd.DataFrame:
    """Add ``r0_clean``, ``strehl_clean``, ``tau0`` and ``seeing`` following Vigan's SPARTA code.

    r0 outside (0, 0.9] m and Strehl outside (0.05, 0.98] become NaN. ``tau0`` is
    ``0.314 * r0 / windspeed`` and ``seeing`` is ``degrees(500e-9 / r0)`` in arcsec. No zenith
    correction is applied.
    """
    out = atm.copy()
    r0 = out["R0"].astype(float)
    strehl = out["StrehlRatio"].astype(float)
    out["r0_clean"] = r0.where((r0 > 0) & (r0 <= 0.9))
    out["strehl_clean"] = strehl.where((strehl > 0.05) & (strehl <= 0.98))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["tau0"] = 0.314 * out["r0_clean"] / out["WindSpeed"].astype(float)
        out["seeing"] = np.degrees(500e-9 / out["r0_clean"]) * 3600.0
    return out


def _dtts_images(hdu, missing: list[str]) -> np.ndarray | None:
    """DTTS images of shape (n, ny, nx), or None (recorded in ``missing``) when the table cannot give them."""
    names = hdu.columns.names
    absent = [c for c in ("Sec", "USec", "WindowSizeX", "WindowSizeY", "Pixels") if c not in names]
    if absent:
        missing.extend(f"{DTTS_EXTENSION}.{c}" for c in absent)
        return None
    data = hdu.data
    ny, nx = int(data["WindowSizeY"][0]), int(data["WindowSizeX"][0])
    pixels = _native(np.asarray(data["Pixels"])).astype(float)
    if ny <= 0 or nx <= 0 or pixels.size != len(data) * ny * nx:
        missing.append(f"{DTTS_EXTENSION}.Pixels")
        return None
    return pixels.reshape(-1, ny, nx)


def _dtts_times(hdu) -> pd.DataFrame:
    start, end = _intervals(DTTS_EXTENSION, _seconds(hdu.data))
    return pd.DataFrame({"mjd_start": start, "mjd_end": end})


def _header_value(header: fits.Header, key: str):
    value = header.get(key)
    if value is None or value == "":
        return None
    return value.strip() if isinstance(value, str) else value


def read_sparta_file(path: Path) -> SpartaFile:
    """Read one SPARTA file, recording missing extensions, columns and header keys instead of raising.

    Rows of a SPARTA table whose values repeat the previous row are dropped, and the sample interval is
    the median spacing of the remaining rows. A table without ``Sec``/``USec`` and a DTTS table whose
    ``Pixels`` do not match its window size are recorded in ``missing`` and read as empty.

    Args:
        path: Path to an uncompressed SPARTA FITS file.

    Returns:
        The parsed ``SpartaFile``. Missing extensions give empty tables and are listed in ``missing``.
    """
    path = Path(path)
    missing: list[str] = []
    with fits.open(path) as hdul:
        config: dict[str, str | int | float | None] = {
            name: _header_value(hdul[0].header, key) for name, key in CONFIG_KEYS.items()
        }
        config["wfs_gain"] = wfs_gain(config["wfs_mode"])  # type: ignore[arg-type]
        for name, value in config.items():
            if value is None:
                missing.append(f"header:{name}")

        tables: dict[str, pd.DataFrame] = {}
        for name, columns in SPARTA_TABLES.items():
            if name not in hdul:
                missing.append(name)
                tables[name] = pd.DataFrame(columns=["mjd_start", "mjd_end", *columns], dtype=float)
                continue
            tables[name] = _read_table(name, hdul[name], columns, missing)

        # A zero or negative flux sample is a dropped WFS frame, not a measurement.
        for name in ("VisLoopParams", "IRLoopParams"):
            tables[name]["Flux_avg"] = tables[name]["Flux_avg"].where(tables[name]["Flux_avg"] > 0)
        vis = tables["VisLoopParams"]
        gain = config["wfs_gain"]
        vis["flux_subap"] = vis["Flux_avg"] / gain / PHOTONS_PER_ADU

        images = np.empty((0, _DEFAULT_DTTS_SIDE, _DEFAULT_DTTS_SIDE))
        dtts = pd.DataFrame(columns=["mjd_start", "mjd_end", "dtts_flux", "dtts_peak"], dtype=float)
        if DTTS_EXTENSION not in hdul:
            missing.append(DTTS_EXTENSION)
        elif hdul[DTTS_EXTENSION].data is not None and len(hdul[DTTS_EXTENSION].data):
            parsed = _dtts_images(hdul[DTTS_EXTENSION], missing)
            if parsed is not None:
                images = parsed
                dtts = pd.concat([_dtts_times(hdul[DTTS_EXTENSION]), dtts_photometry(images)], axis=1)
                # An image without a positive pixel was dropped, not measured. The flux of a measured
                # image without the star scatters around zero and stays a measurement.
                dropped = ~(np.nan_to_num(images, nan=0.0) > 0).any(axis=(1, 2))
                dtts.loc[dropped, ["dtts_flux", "dtts_peak"]] = np.nan

    if missing:
        logger.debug("%s: missing %s", path.name, missing)
    return SpartaFile(
        dp_id=path.name.removesuffix(".fits"), config=config, tables=tables, dtts=dtts, dtts_images=images, missing=missing
    )

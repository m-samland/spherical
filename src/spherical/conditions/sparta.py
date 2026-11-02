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

DEFAULT_CADENCE_S: dict[str, float] = {
    "AtmPerfParams": 20.0,
    "VisLoopParams": 10.0,
    "IRLoopParams": 30.0,
    "IRPixelAvgFrame": 2.0,
}

# Where a time stamp sits in the averaging interval it labels. Determined on the 51 Eri night of
# 2017-09-28 (files 08:18:45, 08:32:37 and 09:14:03) by averaging the 2 s DTTS image sums
# (border-median background removed) over the 30 s (IRLoopParams) or 10-15 s (VisLoopParams) window
# ending at, centred on and starting at each time stamp, and regressing Flux_avg on that mean.
# Relative RMS of the residual, over the three files:
#   IRLoopParams:  end 0.0006-0.0007, mid 0.0026-0.0042, start 0.0044-0.0074
#   VisLoopParams: end 0.0075-0.0096, mid 0.0081-0.0111, start 0.0090-0.0125
# The end convention wins in every file, clearly for IRLoopParams and marginally for VisLoopParams.
# AtmPerfParams has no flux and follows VisLoopParams.
SAMPLE_STAMP: dict[str, str] = {
    "AtmPerfParams": "end",
    "VisLoopParams": "end",
    "IRLoopParams": "end",
}

PHOTONS_PER_ADU = 17.0
_EXPECTED_BORDER = 2
_DEFAULT_DTTS_SIDE = 32


@dataclass
class SpartaFile:
    """Parsed content of one SPARTA file.

    Attributes:
        dp_id: File stem, e.g. ``SPHER.2017-09-28T09:14:03.195``.
        config: Header values from ``CONFIG_KEYS`` (``None`` when absent) plus ``wfs_gain``.
        tables: One DataFrame per entry of ``SPARTA_TABLES`` with ``mjd_start``, ``mjd_end`` and
            the value columns. VisLoopParams also has ``flux_subap``.
        dtts: DTTS photometry per 2 s image with ``mjd_start``, ``mjd_end``, ``dtts_sum`` and
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


def _time_columns(name: str, hdu) -> pd.DataFrame:
    data = hdu.data
    if data is None or len(data) == 0:
        return pd.DataFrame({"mjd_start": np.empty(0), "mjd_end": np.empty(0)})
    seconds = _native(data["Sec"]).astype(float) + _native(data["USec"]).astype(float) * 1e-6
    start, end = _intervals(name, seconds)
    return pd.DataFrame({"mjd_start": start, "mjd_end": end})


def dtts_photometry(images: np.ndarray) -> pd.DataFrame:
    """Background-subtracted sum and peak of each DTTS image.

    The background is the median of the outer 2-pixel border. The peak is the vertex of a 2D
    quadratic fitted by least squares to the 3x3 pixels around the brightest pixel. When that
    quadratic is not negative definite (a flat top) or its vertex leaves the 3x3 window, the fitted
    value at the brightest pixel is used.

    Args:
        images: Array of shape (n, ny, nx).

    Returns:
        DataFrame with columns ``dtts_sum`` and ``dtts_peak``.
    """
    sums = np.empty(len(images))
    peaks = np.empty(len(images))
    yy, xx = np.mgrid[-1:2, -1:2]
    design = np.column_stack([xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel(), xx.ravel(), yy.ravel(), np.ones(9)])
    for i, image in enumerate(np.asarray(images, dtype=float)):
        border = np.concatenate(
            [
                image[:_EXPECTED_BORDER].ravel(),
                image[-_EXPECTED_BORDER:].ravel(),
                image[:, :_EXPECTED_BORDER].ravel(),
                image[:, -_EXPECTED_BORDER:].ravel(),
            ]
        )
        image = image - np.median(border)
        sums[i] = image.sum()
        # The centre of all pixels tied at the maximum keeps a flat-topped block centred.
        tied = np.argwhere(image == image.max())
        row, col = np.clip(np.rint(tied.mean(axis=0)).astype(int), 1, np.array(image.shape) - 2)
        patch = image[row - 1 : row + 2, col - 1 : col + 2].ravel()
        a, b, c, d, e, f = np.linalg.lstsq(design, patch, rcond=None)[0]
        peaks[i] = f
        if a < 0 and 4 * a * b - c * c > 0:
            vertex = np.linalg.solve([[2 * a, c], [c, 2 * b]], [-d, -e])
            if np.all(np.abs(vertex) <= 1.5):
                peaks[i] = f + 0.5 * (d * vertex[0] + e * vertex[1])
    return pd.DataFrame({"dtts_sum": sums, "dtts_peak": peaks})


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


def _header_value(header: fits.Header, key: str):
    value = header.get(key)
    if value is None or value == "":
        return None
    return value.strip() if isinstance(value, str) else value


def read_sparta_file(path: Path) -> SpartaFile:
    """Read one SPARTA file, recording missing extensions, columns and header keys instead of raising.

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
            hdu = hdul[name]
            frame = _time_columns(name, hdu)
            for column in columns:
                if hdu.data is not None and column in hdu.columns.names:
                    frame[column] = _native(hdu.data[column]).astype(float) if len(hdu.data) else np.empty(0)
                else:
                    missing.append(f"{name}.{column}")
                    frame[column] = np.nan
            tables[name] = frame

        vis = tables["VisLoopParams"]
        gain = config["wfs_gain"]
        vis["flux_subap"] = vis["Flux_avg"] / gain / PHOTONS_PER_ADU

        if DTTS_EXTENSION in hdul and hdul[DTTS_EXTENSION].data is not None and len(hdul[DTTS_EXTENSION].data):
            hdu = hdul[DTTS_EXTENSION]
            ny = int(hdu.data["WindowSizeY"][0])
            nx = int(hdu.data["WindowSizeX"][0])
            images = _native(hdu.data["Pixels"]).astype(float).reshape(-1, ny, nx)
            dtts = pd.concat([_time_columns(DTTS_EXTENSION, hdu), dtts_photometry(images)], axis=1)
        else:
            if DTTS_EXTENSION not in hdul:
                missing.append(DTTS_EXTENSION)
            images = np.empty((0, _DEFAULT_DTTS_SIDE, _DEFAULT_DTTS_SIDE))
            dtts = pd.DataFrame(columns=["mjd_start", "mjd_end", "dtts_sum", "dtts_peak"], dtype=float)

    if missing:
        logger.debug("%s: missing %s", path.name, missing)
    return SpartaFile(
        dp_id=path.name.removesuffix(".fits"), config=config, tables=tables, dtts=dtts, dtts_images=images, missing=missing
    )

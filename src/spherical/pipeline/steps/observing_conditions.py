"""Pipeline step that builds the per-frame observing-condition tables and plots them.

The tables come from :func:`spherical.conditions.run.run_conditions`; this module reads the frame
tables, passes the ESO login and draws ``conditions/conditions.png``. The step never raises, so
an archive outage or an unreadable file leaves the rest of the reduction untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from spherical.conditions import COVERAGE_FILE, ConditionsSettings, run_conditions
from spherical.conditions.ambient import parse_ambient_csv
from spherical.conditions.run import INCOMPLETE_COVERAGE_FILE
from spherical.pipeline.pipeline_config import PreprocConfig

STEP_NAME = "observing_conditions"
FRAME_TYPES = ("CORO", "CENTER", "FLUX")
SCIENCE_FRAME_TYPES = ("CORO", "CENTER")
PLOT_FILE = "conditions.png"
# reduction_status reports the last record's status, so an archive outage or a failed plot,
# after which the reduction is still healthy, must not read as "failed".
SOFT_FAILURE = "warning"
CONDITIONS_DIR = "conditions"

UNIX_EPOCH_MJD = 40587.0
SECONDS_PER_DAY = 86400.0
# Frames closer than this are drawn as one shaded block, so a 256-frame sequence is not 256 spans.
MERGE_FRAMES_WITHIN_S = 120.0
PLOT_MARGIN_S = 300.0

FRAME_COLORS = {"FLUX": "#eb6834", "CENTER": "#1baf7a", "CORO": "#2a78d6"}
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SEGMENT_COLOR = "#52514e"


def _to_datetime(mjd) -> pd.DatetimeIndex:
    return pd.to_datetime((np.asarray(mjd, dtype=float) - UNIX_EPOCH_MJD) * SECONDS_PER_DAY, unit="s")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_ambient(sparta_dir: Path, name: str, logger) -> pd.DataFrame:
    path = sparta_dir / f"ambient_{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return parse_ambient_csv(path.read_text())
    except Exception as exc:
        logger.warning(f"Ambient file {path.name} not plotted: {exc}")
        return pd.DataFrame()


def _column(table: pd.DataFrame, prefix: str) -> pd.Series | None:
    """The first column whose name starts with *prefix*; the archive appends units to the names."""
    for name in table.columns:
        if str(name).startswith(prefix):
            return pd.to_numeric(table[name], errors="coerce")
    return None


def _merged_intervals(frames: pd.DataFrame) -> list[tuple[float, float]]:
    if not len(frames):
        return []
    rows = frames[["MJD START", "MJD END"]].astype(float).sort_values("MJD START").to_numpy()
    merged = [list(rows[0])]
    for start, end in rows[1:]:
        if (start - merged[-1][1]) * SECONDS_PER_DAY <= MERGE_FRAMES_WITHIN_S:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _midpoints(table: pd.DataFrame) -> np.ndarray:
    return 0.5 * (table["mjd_start"].to_numpy(dtype=float) + table["mjd_end"].to_numpy(dtype=float))


def _science_median(table: pd.DataFrame, column: str, science: list[tuple[float, float]]) -> float:
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    mid = _midpoints(table)
    inside = np.zeros(len(table), dtype=bool)
    for start, end in science:
        inside |= (mid >= start) & (mid <= end)
    reference = values[inside & np.isfinite(values)]
    if not len(reference):
        reference = values[np.isfinite(values)]
    return float(np.median(reference)) if len(reference) else np.nan


def _plot_sparta(ax, table: pd.DataFrame, column: str, label: str, color: str, scale: float = 1.0) -> bool:
    """Plot one SPARTA column per configuration segment, so no line bridges a segment change."""
    if not len(table) or column not in table:
        return False
    values = pd.to_numeric(table[column], errors="coerce") * scale
    if not np.isfinite(values).any():
        return False
    segments = table["segment"] if "segment" in table else pd.Series(0, index=table.index)
    mid = _midpoints(table)
    first = True
    for segment in pd.unique(segments):
        mask = (segments == segment).to_numpy()
        ax.plot(_to_datetime(mid[mask]), values[mask], color=color, lw=1, label=label if first else None)
        first = False
    return True


def _plot_ambient(ax, table: pd.DataFrame, prefix: str, label: str, color: str, scale: float = 1.0) -> bool:
    values = _column(table, prefix) if len(table) else None
    if values is None or not np.isfinite(values).any():
        return False
    ax.plot(_to_datetime(table["mjd"]), values * scale, color=color, lw=1, marker="o", ms=3, label=label)
    return True


def plot_conditions(output_dir: Path, sparta_dir: Path, frames_info: dict[str, pd.DataFrame], logger) -> Path:
    """Draw the conditions through the sequence into ``output_dir / conditions.png``.

    Panels share a UT axis: flux proxies normalised to their median over the science frames,
    SPARTA Strehl, seeing, tau0, wind speed, sky IR temperature, PWV and the DIMM relative flux
    RMS. FLUX, CENTER and CORO frames are shaded, configuration changes are dashed lines and
    hatched spans are stretches without SPARTA samples.

    Args:
        output_dir: Directory holding the ``sparta_*.csv`` tables and the coverage file.
        sparta_dir: Directory holding the cached ``ambient_<name>.csv`` replies.
        frames_info: Frame tables by frame type with ``MJD START`` and ``MJD END``.
        logger: Logger for unreadable inputs.

    Returns:
        Path of the written plot.
    """
    output_dir, sparta_dir = Path(output_dir), Path(sparta_dir)
    sparta = {name: _read_csv(output_dir / f"sparta_{name}.csv") for name in ("atm", "visloop", "irloop", "dtts", "config")}
    ambient = {name: _read_ambient(sparta_dir, name, logger) for name in ("mass", "dimm", "meteo", "lhatpro")}
    coverage_path = next((p for p in (output_dir / COVERAGE_FILE, output_dir / INCOMPLETE_COVERAGE_FILE) if p.exists()), None)
    gaps = json.loads(coverage_path.read_text()).get("gaps", []) if coverage_path else []

    intervals = {ft: _merged_intervals(frames_info[ft]) for ft in FRAME_TYPES if ft in frames_info}
    science = [iv for ft in SCIENCE_FRAME_TYPES for iv in intervals.get(ft, [])]
    every = [iv for ivs in intervals.values() for iv in ivs]

    titles = [
        "Flux / median",
        "SPARTA Strehl",
        'Seeing ["]',
        "tau0 [ms]",
        "Wind [m/s]",
        "Sky IR temp. [C]",
        "PWV [mm]",
        "DIMM flux RMS",
    ]
    fig = Figure(figsize=(10, 1.7 * len(titles) + 1))
    axes = fig.subplots(len(titles), 1, sharex=True)
    have = [False] * len(titles)

    flux_series = (("visloop", "flux_subap", "VisLoop WFS"), ("irloop", "Flux_avg", "IRLoop"), ("dtts", "dtts_sum", "DTTS sum"))
    for (table_name, column, label), color in zip(flux_series, SERIES_COLORS):
        table = sparta[table_name]
        if len(table) and column in table:
            median = _science_median(table, column, science)
            if np.isfinite(median) and median != 0:
                have[0] |= _plot_sparta(axes[0], table, column, label, color, scale=1.0 / median)
    have[1] = _plot_sparta(axes[1], sparta["atm"], "strehl_clean", "SPARTA", SERIES_COLORS[0])
    have[2] = _plot_sparta(axes[2], sparta["atm"], "seeing", "SPARTA", SERIES_COLORS[0])
    have[2] |= _plot_ambient(axes[2], ambient["mass"], "MASS-DIMM Seeing", "MASS-DIMM", SERIES_COLORS[1])
    have[3] = _plot_sparta(axes[3], sparta["atm"], "tau0", "SPARTA", SERIES_COLORS[0], scale=1e3)
    have[3] |= _plot_ambient(axes[3], ambient["mass"], "MASS-DIMM Tau0", "MASS-DIMM", SERIES_COLORS[1], scale=1e3)
    have[4] = _plot_sparta(axes[4], sparta["atm"], "WindSpeed", "SPARTA", SERIES_COLORS[0])
    have[4] |= _plot_ambient(axes[4], ambient["meteo"], "Wind Speed at 30m", "Meteo 30 m", SERIES_COLORS[1])
    have[5] = _plot_ambient(axes[5], ambient["lhatpro"], "IR temperature", "LHATPRO", SERIES_COLORS[0])
    have[6] = _plot_ambient(axes[6], ambient["lhatpro"], "Precipitable Water Vapour", "LHATPRO", SERIES_COLORS[0])
    have[7] = _plot_ambient(axes[7], ambient["dimm"], "Relative Flux RMS", "DIMM", SERIES_COLORS[0])

    config = sparta["config"]
    changes = []
    if len(config) and "segment" in config:
        ordered = config.sort_values("mjd_start")
        changes = ordered.loc[ordered["segment"].diff().fillna(0) != 0, "mjd_start"].dropna().tolist()

    for ax, title, has_data in zip(axes, titles, have):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, color="#e0e0e0", lw=0.5)
        for frame_type, ivs in intervals.items():
            for start, end in ivs:
                ax.axvspan(*_to_datetime([start, end]), color=FRAME_COLORS[frame_type], alpha=0.12, lw=0)
        for gap in gaps:
            ax.axvspan(*_to_datetime([gap["mjd_start"], gap["mjd_end"]]), facecolor="none", edgecolor="#999999",
                       hatch="///", lw=0)
        for change in changes:
            ax.axvline(_to_datetime([change])[0], color=SEGMENT_COLOR, ls="--", lw=1)
        if not has_data:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="#52514e")
        elif len({ln.get_label() for ln in ax.get_lines() if not ln.get_label().startswith("_")}) > 1:
            ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)

    handles = [Patch(color=FRAME_COLORS[ft], alpha=0.3, label=ft) for ft in FRAME_TYPES if ft in intervals]
    if gaps:
        handles.append(Patch(facecolor="none", edgecolor="#999999", hatch="///", label="no SPARTA samples"))
    if changes:
        handles.append(Line2D([], [], color=SEGMENT_COLOR, ls="--", lw=1, label="WFS configuration change"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9)

    if every:
        lo = min(a for a, _ in every) - PLOT_MARGIN_S / SECONDS_PER_DAY
        hi = max(b for _, b in every) + PLOT_MARGIN_S / SECONDS_PER_DAY
        axes[-1].set_xlim(*_to_datetime([lo, hi]))
        axes[-1].set_xlabel(f"UT on {_to_datetime([lo])[0]:%Y-%m-%d}")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = output_dir / PLOT_FILE
    fig.savefig(path, dpi=120)
    return path


def run_observing_conditions(
    converted_dir: Path,
    sparta_dir: Path,
    conditions: ConditionsSettings,
    preprocessing: PreprocConfig,
    logger,
) -> None:
    """Fetch SPARTA and ambient-monitor data, match them to every frame and plot them.

    Reads the ``frames_info_{coro,center,flux}.csv`` that exist in ``converted_dir`` and writes
    the tables, the coverage file and ``conditions.png`` to ``converted_dir / "conditions"``.
    Raw SPARTA files and ambient replies are cached in ``sparta_dir``. Any failure is logged and
    the step returns, so the reduction continues.

    Args:
        converted_dir: The observation's ``converted/`` directory.
        sparta_dir: Cache for the raw SPARTA files and ambient replies.
        conditions: Step settings, ``config.conditions``.
        preprocessing: ``config.preprocessing``, for the ESO login.
        logger: Pipeline logger.
    """
    converted_dir = Path(converted_dir)
    output_dir = converted_dir / CONDITIONS_DIR
    logger.info("Starting observing conditions step.", extra={"step": STEP_NAME, "status": "started"})
    try:
        frames_info = {
            ft: pd.read_csv(converted_dir / f"frames_info_{ft.lower()}.csv")
            for ft in FRAME_TYPES
            if (converted_dir / f"frames_info_{ft.lower()}.csv").exists()
        }
        if not frames_info:
            logger.error("No frames_info_*.csv found; run compute_frames_info first.",
                         extra={"step": STEP_NAME, "status": "failed"})
            return None
        summary = run_conditions(
            frames_info,
            Path(sparta_dir),
            output_dir,
            conditions,
            logger,
            eso_username=preprocessing.eso_username,
            store_password=preprocessing.store_password,
        )
    except Exception as exc:
        logger.error(f"Observing conditions step failed: {exc}", exc_info=True,
                     extra={"step": STEP_NAME, "status": "failed"})
        return None

    try:
        plot_conditions(output_dir, Path(sparta_dir), frames_info, logger)
    except Exception as exc:
        logger.warning(f"Conditions plot failed: {exc}", exc_info=True, extra={"step": STEP_NAME, "status": SOFT_FAILURE})

    if summary.complete:
        logger.info("Finished observing conditions step.", extra={"step": STEP_NAME, "status": "success"})
    else:
        failed = {k: summary.messages.get(k, "") for k, v in summary.sources.items() if v == "failed"}
        logger.warning(f"Observing conditions incomplete, the next run retries: {failed}",
                       extra={"step": STEP_NAME, "status": SOFT_FAILURE})
    return None

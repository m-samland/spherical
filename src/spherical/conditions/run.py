"""Build per-frame observing-condition tables for one observation and a coverage summary.

Ties together the SPARTA reader, the archive and ambient queries and the frame matching. Network and
data problems are recorded in the `CoverageSummary` and never raised, so the reduction continues.
Only numpy, pandas and astropy are used, so the module imports in a base install.
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from spherical.conditions.ambient import parse_ambient_csv, query_ambient, unavailable_reason
from spherical.conditions.archive import Deadline, SourceFailed, download_sparta_files, find_sparta_files
from spherical.conditions.matching import find_gaps, match_to_frames
from spherical.conditions.sparta import (
    CONFIG_KEYS,
    SPARTA_TABLES,
    SpartaFile,
    derive_atmospheric,
    read_sparta_file,
)

COVERAGE_FILE = "coverage.json"
INCOMPLETE_COVERAGE_FILE = "coverage_incomplete.json"
SPARTA_LIST_CACHE = "sparta_files.csv"
GAP_QUERY_CACHE = "sparta_gap_queries.json"
BUDGET_MESSAGE = "time budget exceeded"
SECONDS_PER_DAY = 86400.0
WAFFLE_COLUMN = "OCS WAFFLE AMPL"
FITS_BLOCK = 2880

CONFIG_SEGMENT_KEYS = ("wfs_mode", "spatial_filter", "wfs_filter", "nd_filter")

# Proxy columns carried into the matched tables, per SPARTA table.
PROXY_COLUMNS: dict[str, tuple[str, ...]] = {
    "visloop": ("flux_subap",),
    "irloop": ("Flux_avg",),
    "dtts": ("dtts_sum", "dtts_peak"),
    "atm": ("strehl_clean", "seeing", "tau0"),
}
_SAMPLE_COLUMNS = ["mjd_start", "mjd_end"]


@dataclass
class ConditionsSettings:
    """Settings of the observing-conditions step."""

    #: Query the archive and ambient monitors again instead of reusing files cached in the SPARTA directory.
    refresh_cache: bool = False
    #: Wall-clock budget in seconds for all network calls of one observation. Sources not reached in time are recorded as failed.
    time_budget_s: float = 120.0
    #: Ambient monitors to query (``mass``, ``dimm``, ``meteo``, ``lhatpro``).
    ambient_sources: tuple[str, ...] = ("mass", "dimm", "meteo", "lhatpro")
    #: Largest distance in seconds between a frame and the neighbouring samples for interpolation across a short hole.
    max_interp_gap_s: float = 60.0
    #: Margin in seconds added on both sides of the frame window for the archive and ambient queries.
    margin_s: float = 600.0


@dataclass
class CoverageSummary:
    """What was found for one observation.

    Attributes:
        sources: Outcome per source, ``found``, ``empty`` or ``failed``. Keys are ``sparta`` and each ambient name.
        messages: Failure or remark per source.
        sparta_files: DP.IDs of the SPARTA files that were read.
        missing_files: DP.IDs the archive listed that are not on disk after the download, for example
            proprietary files without an ESO login. They make the SPARTA source ``failed``.
        unreadable_files: DP.ID and error of complete files on disk that could not be parsed. They are
            kept and not fetched again, so they do not make the source fail.
        missing_tables: Per DP.ID that was read, the extensions, columns and header keys it lacks.
        gaps: Time ranges without SPARTA samples during the frames, each ``{"mjd_start", "mjd_end",
            "unfiltered_files"}``. ``unfiltered_files`` lists ``{"dp_id", "ob_id", "mjd_obs"}`` for each file
            the unfiltered archive query returned for that range that was not listed already, and is
            ``None`` when the query did not run.
        direct_fraction: Per proxy column, the fraction of science frames (CORO and CENTER) with a directly measured, non-NaN value.
    """

    sources: dict[str, str] = field(default_factory=dict)
    messages: dict[str, str] = field(default_factory=dict)
    sparta_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    unreadable_files: dict[str, str] = field(default_factory=dict)
    missing_tables: dict[str, list[str]] = field(default_factory=dict)
    gaps: list[dict] = field(default_factory=list)
    direct_fraction: dict[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether no source failed."""
        return "failed" not in self.sources.values()

    def write(self, output_dir: Path) -> Path:
        """Write ``coverage.json`` when complete, otherwise ``coverage_incomplete.json``.

        A complete summary gates the step, so an incomplete one removes a stale ``coverage.json``
        and the next run retries.

        Args:
            output_dir: Directory for the file, created if missing.

        Returns:
            Path of the written file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        done, pending = output_dir / COVERAGE_FILE, output_dir / INCOMPLETE_COVERAGE_FILE
        target, other = (done, pending) if self.complete else (pending, done)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        other.unlink(missing_ok=True)
        return target

    def to_dict(self) -> dict:
        """Return the summary as a JSON-serialisable dictionary."""
        return {
            "complete": self.complete,
            "sources": self.sources,
            "messages": self.messages,
            "sparta_files": self.sparta_files,
            "missing_files": self.missing_files,
            "unreadable_files": self.unreadable_files,
            "missing_tables": self.missing_tables,
            "gaps": self.gaps,
            "direct_fraction": {k: (None if np.isnan(v) else v) for k, v in self.direct_fraction.items()},
        }


def _sparta_path(sparta_dir: Path, dp_id: str) -> Path | None:
    for suffix in (".fits", ".fits.Z"):
        candidate = sparta_dir / f"{dp_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _readable_path(path: Path) -> Path:
    """Return an uncompressed copy of a ``.fits.Z`` file next to it, creating it on first use."""
    if path.suffix != ".Z":
        return path
    target = path.with_suffix("")
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        try:
            # Same convention as the raw-data download step: gzip first, then the system uncompress for true LZW files.
            try:
                with gzip.open(path, "rb") as fin, open(partial, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            except OSError:
                if shutil.which("uncompress") is None:
                    raise FileNotFoundError("the uncompress tool is not installed") from None
                with open(path, "rb") as fin, open(partial, "wb") as fout:
                    subprocess.run(["uncompress", "-c"], stdin=fin, stdout=fout, check=True)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return target


def _frame_intervals(frames_info: dict[str, pd.DataFrame], join_s: float) -> list[tuple[float, float]]:
    """Exposure intervals of all frames, merged where frames are less than ``join_s`` apart."""
    rows = sorted(
        (float(a), float(b))
        for t in frames_info.values()
        if len(t)
        for a, b in zip(t["MJD START"], t["MJD END"], strict=True)
        if np.isfinite(a) and np.isfinite(b)
    )
    merged: list[list[float]] = []
    for start, end in rows:
        if merged and (start - merged[-1][1]) * SECONDS_PER_DAY <= join_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _within_frames(lo: float, hi: float, intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(max(lo, a), min(hi, b)) for a, b in intervals if min(hi, b) > max(lo, a)]


def _frame_window(frames_info: dict[str, pd.DataFrame]) -> tuple[list[int], float, float]:
    tables = [t for t in frames_info.values() if len(t)]
    ob_ids = sorted({int(v) for t in tables if "OBS ID" in t for v in t["OBS ID"].dropna()})
    start = min(float(t["MJD START"].min()) for t in tables)
    end = max(float(t["MJD END"].max()) for t in tables)
    return ob_ids, start, end


def _list_sparta_files(ob_ids, window, sparta_dir, settings, deadline, summary, logger) -> list[str]:
    cache = sparta_dir / SPARTA_LIST_CACHE
    if cache.exists() and not settings.refresh_cache:
        try:
            return list(pd.read_csv(cache)["dp_id"])
        except Exception as exc:
            logger.warning("Ignoring unreadable cache %s: %s", cache, exc)
    if deadline.expired:
        summary.sources["sparta"], summary.messages["sparta"] = "failed", BUDGET_MESSAGE
        return []
    try:
        listing = find_sparta_files(ob_ids, *window, deadline=deadline)
    except SourceFailed as exc:
        logger.warning("SPARTA file query failed: %s", exc)
        summary.sources["sparta"], summary.messages["sparta"] = "failed", str(exc)
        return []
    if len(listing):
        listing.to_csv(cache, index=False)
    return list(listing["dp_id"])


def _fetch_sparta_files(dp_ids, sparta_dir, deadline, summary, logger, eso_username, store_password) -> None:
    if all(_sparta_path(sparta_dir, d) is not None for d in dp_ids):
        return
    if deadline.expired:
        summary.sources["sparta"], summary.messages["sparta"] = "failed", BUDGET_MESSAGE
        return
    try:
        download_sparta_files(dp_ids, sparta_dir, deadline=deadline, eso_username=eso_username, store_password=store_password)
    except SourceFailed as exc:
        logger.warning("SPARTA download failed: %s", exc)
        summary.sources["sparta"], summary.messages["sparta"] = "failed", str(exc)


def _fail_sparta(summary: CoverageSummary, note: str) -> None:
    summary.sources["sparta"] = "failed"
    summary.messages["sparta"] = "; ".join(filter(None, [summary.messages.get("sparta"), note]))


def _discard(path: Path) -> None:
    for leftover in (path, path.with_suffix(""), path.with_name(path.with_suffix("").name + ".part")):
        leftover.unlink(missing_ok=True)


def _read_files(dp_ids, sparta_dir, logger, summary) -> list[SpartaFile]:
    """Read the listed files that are on disk.

    Only a truncated file is deleted, so that the next run fetches it again. A complete file that does
    not parse is kept and recorded in ``summary.unreadable_files``, because a new download would give
    the same bytes.
    """
    files = []
    for dp_id in dp_ids:
        path = _sparta_path(sparta_dir, dp_id)
        if path is None:
            logger.warning("SPARTA file %s is not on disk", dp_id)
            summary.missing_files.append(dp_id)
            continue
        try:
            readable = _readable_path(path)
        except (EOFError, subprocess.CalledProcessError) as exc:
            logger.warning("Removing truncated SPARTA file %s: %s", path.name, exc)
            _discard(path)
            _fail_sparta(summary, f"truncated SPARTA file {dp_id} removed for re-download")
            continue
        except Exception as exc:
            logger.warning("Could not decompress SPARTA file %s: %s", path.name, exc)
            _fail_sparta(summary, f"could not decompress SPARTA file {dp_id}: {exc}")
            continue
        if _truncated(readable):
            logger.warning("Removing truncated SPARTA file %s", path.name)
            _discard(path)
            _fail_sparta(summary, f"truncated SPARTA file {dp_id} removed for re-download")
            continue
        try:
            parsed = read_sparta_file(readable)
            if all(name in parsed.missing for name in SPARTA_TABLES):
                raise ValueError("no SPARTA table found")
        except Exception as exc:
            logger.warning("Keeping unreadable SPARTA file %s: %s", path.name, exc)
            summary.unreadable_files[dp_id] = str(exc)
            continue
        parsed.dp_id = dp_id
        if parsed.missing:
            logger.info("SPARTA file %s lacks %s", dp_id, parsed.missing)
            summary.missing_tables[dp_id] = list(parsed.missing)
        files.append(parsed)
    return files


def _truncated(path: Path) -> bool:
    """Whether a download stopped early: a FITS file is whole 2880-byte blocks and holds all of its last HDU."""
    size = path.stat().st_size
    if size == 0 or size % FITS_BLOCK:
        return True
    try:
        with fits.open(path) as hdul:
            info = hdul[-1].fileinfo()
            return info is not None and info["datLoc"] + info["datSpan"] > size
    except Exception:
        # A whole number of blocks that astropy cannot open is a corrupt file; the reader reports it.
        return False


def _file_range(file: SpartaFile) -> tuple[float, float]:
    tables = [*file.tables.values(), file.dtts]
    starts = [t["mjd_start"].min() for t in tables if len(t)]
    ends = [t["mjd_end"].max() for t in tables if len(t)]
    return (float(min(starts)), float(max(ends))) if starts else (np.nan, np.nan)


def _assign_segments(files: list[SpartaFile]) -> tuple[list[SpartaFile], list[int], list[tuple[float, float]]]:
    ranges = {f.dp_id: _file_range(f) for f in files}
    ordered = sorted(files, key=lambda f: (np.isnan(ranges[f.dp_id][0]), ranges[f.dp_id][0], f.dp_id))
    segments, current, previous = [], -1, None
    for file in ordered:
        key = tuple(file.config.get(k) for k in CONFIG_SEGMENT_KEYS)
        if key != previous:
            current += 1
            previous = key
        segments.append(current)
    return ordered, segments, [ranges[f.dp_id] for f in ordered]


def _concat(files, segments, pick) -> pd.DataFrame:
    parts = []
    for file, segment in zip(files, segments, strict=True):
        table = pick(file)
        if len(table):
            parts.append(table.assign(dp_id=file.dp_id, segment=segment))
    if not parts:
        return pd.DataFrame({"mjd_start": [], "mjd_end": [], "segment": np.empty(0, dtype=int)})
    return pd.concat(parts, ignore_index=True).sort_values("mjd_start", kind="stable").reset_index(drop=True)


def _config_table(files, segments, ranges, frames_info) -> pd.DataFrame:
    rows = []
    for file, segment, (start, end) in zip(files, segments, ranges, strict=True):
        amplitudes = []
        for table in frames_info.values():
            if WAFFLE_COLUMN in table and not np.isnan(start):
                mid = 0.5 * (table["MJD START"] + table["MJD END"])
                amplitudes.append(pd.to_numeric(table.loc[(mid >= start) & (mid <= end), WAFFLE_COLUMN], errors="coerce"))
        amplitude = pd.concat(amplitudes).median() if amplitudes else np.nan
        rows.append(
            {
                "dp_id": file.dp_id,
                **{key: file.config.get(key) for key in CONFIG_KEYS},
                "wfs_gain": file.config.get("wfs_gain"),
                "mjd_start": start,
                "mjd_end": end,
                "segment": segment,
                "waffle_amplitude": amplitude,
            }
        )
    columns = ["dp_id", *CONFIG_KEYS, "wfs_gain", "mjd_start", "mjd_end", "segment", "waffle_amplitude"]
    return pd.DataFrame(rows, columns=columns)


def _hits(found: pd.DataFrame) -> list[dict]:
    return [
        {
            "dp_id": str(row.dp_id),
            "ob_id": None if pd.isna(row.ob_id) else int(row.ob_id),
            "mjd_obs": None if pd.isna(row.mjd_obs) else float(row.mjd_obs),
        }
        for row in found.itertuples(index=False)
    ]


def _find_gaps_with_unfiltered_query(samples, window, frames, known, sparta_dir, settings, deadline, summary, logger) -> None:
    cache_path = sparta_dir / GAP_QUERY_CACHE
    cache: dict[str, list[dict]] = {}
    if cache_path.exists() and not settings.refresh_cache:
        try:
            cache = {
                k: v
                for k, v in json.loads(cache_path.read_text()).items()
                if isinstance(v, list) and all(isinstance(h, dict) for h in v)
            }
        except Exception:
            cache = {}
    # A file's mjd_obs is when it was written, after its samples, so a file covering the gap can start past its end.
    margin = settings.margin_s / SECONDS_PER_DAY
    holes = [piece for lo, hi in find_gaps(samples, window) for piece in _within_frames(lo, hi, frames)]
    for lo, hi in holes:
        # The pause between consecutive SPARTA files (20 to 30 s) is not an archive hole worth a query.
        if (hi - lo) * SECONDS_PER_DAY < settings.max_interp_gap_s:
            continue
        key = f"{lo:.6f}_{hi:.6f}"
        hits = cache.get(key)
        if hits is None:
            if summary.sources.get("sparta") == "failed":
                pass
            elif deadline.expired:
                summary.sources["sparta"], summary.messages["sparta"] = "failed", BUDGET_MESSAGE
            else:
                try:
                    hits = _hits(find_sparta_files([], lo, hi + margin, deadline=deadline, filtered=False))
                    cache[key] = hits
                except SourceFailed as exc:
                    logger.warning("Unfiltered SPARTA query for a gap failed: %s", exc)
                    summary.sources["sparta"], summary.messages["sparta"] = "failed", str(exc)
        if hits is not None:
            hits = [h for h in hits if h["dp_id"] not in known]
        logger.info(
            "No SPARTA samples for %.1f min (MJD %.5f to %.5f); unfiltered query found %s",
            (hi - lo) * 1440,
            lo,
            hi,
            "nothing (not run)" if hits is None else [h["dp_id"] for h in hits],
        )
        summary.gaps.append({"mjd_start": lo, "mjd_end": hi, "unfiltered_files": hits})
    if cache:
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError as exc:
            logger.warning("Could not cache gap queries: %s", exc)


def _query_ambient_sources(window, sparta_dir, settings, deadline, summary, logger) -> None:
    for name in settings.ambient_sources:
        reason = unavailable_reason(name, *window)
        if reason:
            summary.sources[name], summary.messages[name] = "empty", reason
            continue
        cache = sparta_dir / f"ambient_{name}.csv"
        text = None
        if cache.exists() and not settings.refresh_cache:
            text = cache.read_text()
        else:
            if deadline.expired:
                summary.sources[name], summary.messages[name] = "failed", BUDGET_MESSAGE
                continue
            try:
                text = query_ambient(name, *window, deadline=deadline)
            except SourceFailed as exc:
                logger.warning("Ambient query %s failed: %s", name, exc)
                summary.sources[name], summary.messages[name] = "failed", str(exc)
                continue
        try:
            rows = parse_ambient_csv(text)
        except SourceFailed as exc:
            logger.warning("Ambient reply %s unusable: %s", name, exc)
            summary.sources[name], summary.messages[name] = "failed", str(exc)
            continue
        summary.sources[name] = "found" if len(rows) else "empty"
        if len(rows):
            cache.write_text(text)


def _match_all(frames_info, tables, max_interp_gap_s) -> dict[str, pd.DataFrame]:
    matched = {}
    for frame_type, frames in frames_info.items():
        columns: dict[str, np.ndarray] = {
            "MJD START": frames["MJD START"].to_numpy(dtype=float),
            "MJD END": frames["MJD END"].to_numpy(dtype=float),
        }
        segment = np.full(len(frames), -1, dtype=int)
        for table_name, value_columns in PROXY_COLUMNS.items():
            result = match_to_frames(tables[table_name], frames, value_columns, max_interp_gap_s=max_interp_gap_s)
            for col in value_columns:
                columns[f"{table_name}_{col}"] = result[col].to_numpy()
                columns[f"{table_name}_{col}_std"] = result[f"{col}_std"].to_numpy()
            columns[f"n_samples_{table_name}"] = result["n_samples"].to_numpy()
            columns[f"flag_{table_name}"] = result["flag"].to_numpy()
            segment = np.where(segment < 0, result["segment"].to_numpy(), segment)
        columns["segment"] = segment
        matched[frame_type] = pd.DataFrame(columns)
    return matched


SCIENCE_FRAME_TYPES = ("CORO", "CENTER")


def _direct_fractions(matched: dict[str, pd.DataFrame]) -> dict[str, float]:
    science = [m for name, m in matched.items() if name.upper() in SCIENCE_FRAME_TYPES]
    total = sum(len(m) for m in science)
    fractions = {}
    for table_name, value_columns in PROXY_COLUMNS.items():
        for col in value_columns:
            direct = sum(
                int(((m[f"flag_{table_name}"] == "direct") & m[f"{table_name}_{col}"].notna()).sum()) for m in science
            )
            fractions[f"{table_name}_{col}"] = direct / total if total else float("nan")
    return fractions


def _build(frames_info, sparta_dir, output_dir, settings, logger, summary, eso_username, store_password) -> None:
    if not any(len(t) for t in frames_info.values()):
        logger.warning("The frame tables hold no frames, so there is nothing to match.")
        for name in ("sparta", *settings.ambient_sources):
            summary.sources[name], summary.messages[name] = "empty", "the frame tables hold no frames"
        return
    deadline = Deadline(settings.time_budget_s)
    ob_ids, science_start, science_end = _frame_window(frames_info)
    margin = settings.margin_s / SECONDS_PER_DAY
    window = (science_start - margin, science_end + margin)

    if ob_ids:
        dp_ids = _list_sparta_files(ob_ids, window, sparta_dir, settings, deadline, summary, logger)
    else:
        dp_ids = []
        summary.sources["sparta"], summary.messages["sparta"] = "empty", "no OB ID in the frame tables to match SPARTA files"
    if dp_ids and summary.sources.get("sparta") != "failed":
        _fetch_sparta_files(dp_ids, sparta_dir, deadline, summary, logger, eso_username, store_password)
    files = _read_files(dp_ids, sparta_dir, logger, summary) if dp_ids else []
    if summary.missing_files and summary.sources.get("sparta") != "failed":
        _fail_sparta(summary, f"listed SPARTA files not on disk: {', '.join(summary.missing_files)}")
    files, segments, ranges = _assign_segments(files)
    summary.sparta_files = [f.dp_id for f in files]
    if "sparta" not in summary.sources:
        summary.sources["sparta"] = "found" if files else "empty"
        if dp_ids and not files:
            summary.messages["sparta"] = "no listed SPARTA file could be read"

    tables = {
        "atm": derive_atmospheric(_concat(files, segments, lambda f: f.tables["AtmPerfParams"]))
        if files
        else pd.DataFrame({**{c: [] for c in _SAMPLE_COLUMNS}, "segment": np.empty(0, dtype=int)}),
        "visloop": _concat(files, segments, lambda f: f.tables["VisLoopParams"]),
        "irloop": _concat(files, segments, lambda f: f.tables["IRLoopParams"]),
        "dtts": _concat(files, segments, lambda f: f.dtts),
    }
    for table_name, value_columns in PROXY_COLUMNS.items():
        for col in value_columns:
            if col not in tables[table_name]:
                tables[table_name][col] = np.nan

    samples = pd.concat([tables["visloop"][_SAMPLE_COLUMNS], tables["dtts"][_SAMPLE_COLUMNS]], ignore_index=True)
    _find_gaps_with_unfiltered_query(
        samples,
        (science_start, science_end),
        _frame_intervals(frames_info, settings.max_interp_gap_s),
        set(dp_ids),
        sparta_dir,
        settings,
        deadline,
        summary,
        logger,
    )
    _query_ambient_sources(window, sparta_dir, settings, deadline, summary, logger)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables["atm"].to_csv(output_dir / "sparta_atm.csv", index=False)
    tables["visloop"].to_csv(output_dir / "sparta_visloop.csv", index=False)
    tables["irloop"].to_csv(output_dir / "sparta_irloop.csv", index=False)
    tables["dtts"].to_csv(output_dir / "sparta_dtts.csv", index=False)
    _config_table(files, segments, ranges, frames_info).to_csv(output_dir / "sparta_config.csv", index=False)

    matched = _match_all(frames_info, tables, settings.max_interp_gap_s)
    for frame_type, table in matched.items():
        table.to_csv(output_dir / f"matched_{frame_type.lower()}.csv", index=False)
    summary.direct_fraction = _direct_fractions(matched)


def run_conditions(
    frames_info: dict[str, pd.DataFrame],
    sparta_dir: Path,
    output_dir: Path,
    settings: ConditionsSettings,
    logger: logging.Logger,
    *,
    eso_username: str | None = None,
    store_password: bool = False,
) -> CoverageSummary:
    """Build the per-frame condition tables and the coverage summary of one observation.

    Data and network problems are logged and recorded in the summary, never raised. Raw SPARTA files,
    the file list and the ambient replies are cached in ``sparta_dir`` and reused unless
    ``settings.refresh_cache`` is set.

    Args:
        frames_info: Frame tables by frame type, each with ``MJD START``, ``MJD END`` and ``OBS ID``
            (and optionally ``OCS WAFFLE AMPL``).
        sparta_dir: Cache directory for the raw SPARTA files and ambient replies.
        output_dir: Directory for ``sparta_*.csv``, ``matched_<frame>.csv`` and the coverage file.
        settings: Step settings.
        logger: Logger for warnings and progress.
        eso_username: ESO user name for the SPARTA download.
        store_password: Let astroquery keep the ESO password in the keyring.

    Returns:
        The coverage summary, already written to ``output_dir``.
    """
    sparta_dir, output_dir = Path(sparta_dir), Path(output_dir)
    summary = CoverageSummary()
    try:
        sparta_dir.mkdir(parents=True, exist_ok=True)
        _build(frames_info, sparta_dir, output_dir, settings, logger, summary, eso_username, store_password)
    except Exception as exc:
        logger.warning("Observing-conditions step failed: %s", exc, exc_info=True)
        summary.sources["sparta"] = "failed"
        summary.messages["sparta"] = f"unexpected error: {exc}"
    for name in settings.ambient_sources:
        if name not in summary.sources:
            summary.sources[name], summary.messages[name] = "failed", summary.messages.get("sparta", "not queried")
    summary.write(output_dir)
    return summary

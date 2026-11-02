"""ESO ambient-monitor queries (MASS-DIMM, DIMM, meteo, LHATPRO) returning CSV text.

Only the standard library, astropy and pandas are used, so the module imports in a base install.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd
from astropy.time import Time

from spherical.conditions.archive import Deadline, SourceFailed

BASE_URL = "https://archive.eso.org/wdb/wdb/asm/{name}_paranal/query"

# SLODAR is left out: it returns a handful of rows per night and says nothing about transmission.
AMBIENT_SOURCES: dict[str, str] = {
    "mass": "tab_fl=1&tab_tau0=1&tab_theta0=1&tab_cn2_gf=1&tab_seeing=1",
    "dimm": "tab_fl=1&tab_rfl=1&tab_rfl_time=1",
    "meteo": "tab_press=0&tab_presqnh=0&tab_temp=1&tab_temp1=0&tab_temp2=0&tab_temp3=0&tab_temp4=0"
    "&tab_dew=0&tab_rhum=0&tab_wind=1&tab_wind_dir=1&tab_wind_dir_30m=0&tab_wind_30m=1",
    "lhatpro": "",
}


def _isot(mjd: float) -> str:
    return Time(mjd, format="mjd", scale="utc").isot.split(".")[0]


def query_ambient(name: str, mjd_start: float, mjd_end: float, *, deadline: Deadline) -> str:
    """Fetch one ambient-monitor table as CSV text.

    Args:
        name: Key of `AMBIENT_SOURCES`.
        mjd_start: Start of the window in MJD.
        mjd_end: End of the window in MJD.
        deadline: Shared time budget.

    Returns:
        The raw CSV reply.

    Raises:
        SourceFailed: On timeout, an HTTP error, an unknown name or an expired deadline.
    """
    deadline.check()
    if name not in AMBIENT_SOURCES:
        raise SourceFailed(f"unknown ambient source {name!r}")
    url = (
        f"{BASE_URL.format(name=name)}?wdbo=csv&start_date={_isot(mjd_start)}..{_isot(mjd_end)}"
        f"&{AMBIENT_SOURCES[name]}"
    ).rstrip("&")
    try:
        with urllib.request.urlopen(url, timeout=min(30.0, max(deadline.remaining(), 0.1))) as reply:
            return reply.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise SourceFailed(f"ambient query {name} failed: {exc}") from exc


def parse_ambient_csv(text: str) -> pd.DataFrame:
    """Parse an ambient-monitor CSV reply.

    Args:
        text: Raw reply from `query_ambient`.

    Returns:
        Table with the reply's columns plus ``mjd`` parsed from ``Date time``. Empty if the reply
        holds no data rows.

    Raises:
        SourceFailed: If the reply has a header but no ``Date time`` column.
    """
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(io.StringIO("\n".join(lines)), skipinitialspace=True)
        frame.columns = [str(c).strip() for c in frame.columns]
        frame = frame.dropna(subset=["Date time"])
        frame["mjd"] = Time(frame["Date time"].astype(str).str.strip().tolist(), format="isot", scale="utc").mjd
    except Exception as exc:
        raise SourceFailed(f"unparsable ambient reply: {exc}") from exc
    return frame.reset_index(drop=True)

"""ESO archive access for SPARTA files: TAP queries, downloads and a shared time budget.

Every network function checks the deadline first and reports any failure as `SourceFailed`,
so a caller can record the outage and continue the reduction.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TAP_URL = "https://archive.eso.org/tap_obs/sync"
TAP_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 60.0
SPARTA_DP_TYPE = "OBJECT,AO"
# astroquery 0.4.x stores and looks up the ESO password under this keyring service name.
KEYRING_SERVICE = "astroquery:www.eso.org"


class SourceFailed(Exception):
    """A network source timed out, returned an error or an unparsable reply, or the time budget ran out.

    Args:
        message: Description of the failure.
        present: For downloads, the paths of the requested files that are on disk despite the failure.
    """

    def __init__(self, message: str = "", present: list | None = None):
        super().__init__(message)
        self.present = list(present) if present else []


class Deadline:
    """Wall-clock budget shared by all network calls of one observation.

    Args:
        seconds: Budget in seconds, counted from construction.
    """

    def __init__(self, seconds: float):
        self._end = time.monotonic() + float(seconds)

    def remaining(self) -> float:
        """Return the seconds left, never negative."""
        return max(0.0, self._end - time.monotonic())

    @property
    def expired(self) -> bool:
        """Whether the budget is used up."""
        return self.remaining() <= 0.0

    def check(self) -> None:
        """Raise `SourceFailed` if the budget is used up."""
        if self.expired:
            raise SourceFailed("time budget exhausted")


def sparta_query(ob_ids: Sequence[int], mjd_start: float, mjd_end: float, *, filtered: bool = True) -> str | None:
    """Build the ADQL query for the SPARTA files of an observation.

    The window is in MJD and the filter lists every OB ID, so an observation that spans several OBs
    or crosses UT midnight is still covered. With ``filtered=False`` only the instrument and time
    window are constrained, which lets a caller detect metadata that does not match the filter.

    Args:
        ob_ids: OB IDs found in the frame tables.
        mjd_start: Start of the window in MJD.
        mjd_end: End of the window in MJD.
        filtered: Restrict to ``dp_type='OBJECT,AO'`` and the given OB IDs.

    Returns:
        The ADQL query string, or ``None`` for a filtered query without OB IDs, which has nothing to match.
    """
    base = "SELECT dp_id, mjd_obs, ob_id, dp_type FROM dbo.raw WHERE instrument='SPHERE'"
    window = f"mjd_obs BETWEEN {float(mjd_start)} AND {float(mjd_end)}"
    if not filtered:
        return f"{base} AND {window}"
    if not len(ob_ids):
        return None
    ids = ", ".join(str(int(i)) for i in ob_ids)
    return f"{base} AND dp_type='{SPARTA_DP_TYPE}' AND ob_id IN ({ids}) AND {window}"


def find_sparta_files(
    ob_ids: Sequence[int],
    mjd_start: float,
    mjd_end: float,
    *,
    deadline: Deadline,
    filtered: bool = True,
) -> pd.DataFrame:
    """Query the ESO TAP sync endpoint over plain HTTP so the time budget is a real socket timeout for SPARTA files.

    Args:
        ob_ids: OB IDs found in the frame tables.
        mjd_start: Start of the window in MJD.
        mjd_end: End of the window in MJD.
        deadline: Shared time budget.
        filtered: See `sparta_query`. The unfiltered form keeps rows whose ``dp_type`` is
            ``OBJECT,AO`` or empty client-side, so a SPARTA file with a wrong OB ID or a missing data
            type shows up as an extra row.

    Returns:
        Table with columns ``dp_id``, ``mjd_obs``, ``ob_id`` and ``dp_type``, sorted by ``mjd_obs``.
        Empty without a network call when `sparta_query` has no query.

    Raises:
        SourceFailed: On timeout, a service error or an expired deadline.
    """
    query = sparta_query(ob_ids, mjd_start, mjd_end, filtered=filtered)
    if query is None:
        return pd.DataFrame({"dp_id": [], "mjd_obs": [], "ob_id": [], "dp_type": []})
    deadline.check()
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    }
    url = f"{TAP_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=min(TAP_TIMEOUT, max(deadline.remaining(), 0.1))) as reply:
            text = reply.read().decode("utf-8", errors="replace")
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = [c.lower() for c in frame.columns]
        frame = frame[["dp_id", "mjd_obs", "ob_id", "dp_type"]].copy()
    except Exception as exc:
        raise SourceFailed(f"SPARTA TAP query failed: {exc}") from exc
    dp_type = frame["dp_type"].fillna("").astype(str).str.strip()
    keep = dp_type == SPARTA_DP_TYPE
    if not filtered:
        keep |= dp_type == ""
    frame = frame[keep]
    return frame.sort_values("mjd_obs").reset_index(drop=True)


def _present(destination: Path, dp_id: str) -> Path | None:
    for suffix in (".fits", ".fits.Z"):
        candidate = destination / f"{dp_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def download_sparta_files(
    dp_ids: Sequence[str],
    destination: Path,
    *,
    deadline: Deadline,
    eso_username: str | None = None,
    store_password: bool = False,
) -> list[Path]:
    """Download SPARTA files that are not yet in ``destination``, without ever prompting for a password.

    The files are fetched anonymously first. Files still missing afterwards (proprietary data
    answers 401) are fetched again after a login, but only when ``eso_username`` is set and its
    password is already stored in the keyring that astroquery uses. The files are kept compressed
    (``.fits.Z``) because they are small and astropy reads them directly.

    Args:
        dp_ids: DP.IDs to fetch.
        destination: Directory for the files, created if missing.
        deadline: Shared time budget.
        eso_username: ESO user name for a login, as in the raw-data download step.
        store_password: Accepted for symmetry with the raw-data download step. The step never prompts,
            so there is no new password to store.

    Returns:
        Paths of all requested files, all of which exist afterwards.

    Raises:
        SourceFailed: On a download error, a requested file that is still not on disk, or an expired
            deadline. Its ``present`` attribute lists the requested files that are on disk, including
            ones fetched before the failure.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    missing = [d for d in dp_ids if _present(destination, d) is None]
    if not missing:
        return _existing(destination, dp_ids)
    try:
        from astroquery.eso import Eso

        eso = Eso()
    except Exception as exc:
        raise SourceFailed(f"SPARTA download failed: {exc}", present=_existing(destination, dp_ids)) from exc
    _retrieve(eso, missing, destination, deadline, dp_ids)
    missing = [d for d in missing if _present(destination, d) is None]
    note = "proprietary SPARTA files need config.preprocessing.eso_username with a password stored in the keyring"
    if missing and eso_username:
        if _login_without_prompt(eso, eso_username):
            _retrieve(eso, missing, destination, deadline, dp_ids)
            missing = [d for d in missing if _present(destination, d) is None]
            note = f"the ESO login as {eso_username} did not give access"
        else:
            note = f"no ESO password for {eso_username} is stored in the keyring ({KEYRING_SERVICE}), so no login was tried"
    if missing:
        raise SourceFailed(
            f"{len(missing)} SPARTA file(s) not downloaded: {', '.join(missing)}; {note}",
            present=_existing(destination, dp_ids),
        )
    return _existing(destination, dp_ids)


def _login_without_prompt(eso, username: str) -> bool:
    """Log in only if the password is already in the keyring; astroquery would otherwise call getpass."""
    try:
        import keyring

        password = keyring.get_password(KEYRING_SERVICE, username)
    except Exception as exc:
        logger.warning("Could not read the keyring for the ESO login: %s", exc)
        return False
    if not password:
        return False
    try:
        return bool(eso.login(username=username, store_password=False))
    except Exception as exc:
        logger.warning("ESO login as %s failed: %s", username, exc)
        return False


def _retrieve(eso, dp_ids: Sequence[str], destination: Path, deadline: Deadline, requested: Sequence[str]) -> None:
    """Fetch files one by one so the deadline is checked between them.

    astroquery logs and swallows a failed file (including a 401), so the caller compares what is on
    disk with what was requested afterwards.
    """
    for dp_id in dp_ids:
        try:
            deadline.check()
            _set_socket_timeout(eso, min(DOWNLOAD_TIMEOUT, deadline.remaining()))
            logger.info("Retrieving SPARTA file %s", dp_id)
            eso.retrieve_data(datasets=[dp_id], destination=str(destination), with_calib=None, unzip=False)
        except Exception as exc:
            raise SourceFailed(f"SPARTA download failed: {exc}", present=_existing(destination, requested)) from exc


def _existing(destination: Path, dp_ids: Sequence[str]) -> list[Path]:
    return [p for p in (_present(destination, d) for d in dp_ids) if p is not None]


def _set_socket_timeout(eso, seconds: float) -> None:
    """Give every request of the Eso session a socket timeout.

    astroquery 0.4.x passes no timeout to its streaming download, so a stalled connection would block
    forever. The timeout bounds each socket read, not the whole file; the deadline is checked between files.
    """
    session = eso._session
    original = getattr(session, "_spherical_original_request", session.request)
    session._spherical_original_request = original

    def request(*args, **kwargs):
        kwargs["timeout"] = kwargs.get("timeout") or max(seconds, 0.1)
        return original(*args, **kwargs)

    session.request = request

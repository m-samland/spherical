"""Choice of SIMBAD server, falling back to the mirrors astroquery knows.

SIMBAD runs at CDS (``simbad.cds.unistra.fr``) and at a Harvard mirror
(``simbad.harvard.edu``). When one is unreachable the other usually still answers, so
every SIMBAD query in spherical goes to the first server that reports itself available.
The server tried first is the one astroquery's shared ``Simbad`` uses, so
``Simbad.server = "simbad.harvard.edu"`` or the ``[simbad]`` section of
``astroquery.cfg`` sets the preference.
"""

import functools
import logging
import warnings
from collections.abc import Iterator

import requests
from astroquery.simbad import Simbad, SimbadClass, conf

logger = logging.getLogger(__name__)

#: Seconds to wait for a server's VOSI availability endpoint. An unreachable server
#: otherwise holds the build until the operating system gives up on the connection.
AVAILABILITY_TIMEOUT = 10.0


def simbad_servers() -> list[str]:
    """Known SIMBAD servers in the order to try them, ``Simbad.server`` first."""
    first = Simbad.server
    return [first] + [server for server in conf.servers_list if server != first]


# Cached for the session, so an unreachable server costs one timeout and not one per
# name lookup.
@functools.cache
def is_available(server: str, timeout: float = AVAILABILITY_TIMEOUT) -> bool:
    """Whether a SIMBAD server's TAP service reports itself available."""
    url = f"https://{server}/simbad/sim-tap/availability"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("SIMBAD server %s is unreachable: %s", server, exc)
        return False
    available = response.ok and "<available>true</available>" in response.text
    if not available:
        logger.warning("SIMBAD server %s is not available (HTTP %s).", server, response.status_code)
    return available


def available_simbad_servers() -> Iterator[str]:
    """Known SIMBAD servers that report themselves available, in the order to try them.

    Lazy, so a server further down the list is only checked once the ones before it fail.
    """
    return (server for server in simbad_servers() if is_available(server))


def simbad_client(server: str) -> SimbadClass:
    """A SIMBAD client bound to ``server``, leaving astroquery's shared ``Simbad`` untouched."""
    client = SimbadClass()
    client.server = server
    return client


def query_object(name: str):
    """Resolve ``name`` on the first SIMBAD server that answers.

    Returns:
        The astroquery result table, or ``None`` when no server could be queried.
    """
    for server in available_simbad_servers():
        try:
            return simbad_client(server).query_object(name)
        except Exception as exc:
            logger.warning("SIMBAD query for %r on %s failed: %s", name, server, exc)
    warnings.warn(f"No SIMBAD server could resolve {name!r}; tried {simbad_servers()}.")
    return None

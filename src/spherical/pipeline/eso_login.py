"""ESO archive login for the download step.

astroquery takes the ESO password from the system keyring and otherwise asks for it. On a
machine without a usable keyring, typical for compute nodes reached over SSH, it cannot store
the password, so every observation that downloads asked again, and the observing-conditions
step, which never asks, could not fetch proprietary SPARTA files. :func:`login_to_eso` keeps
the password in memory for the rest of the process on such machines. Machines with a working
keyring behave as astroquery does.
"""

from __future__ import annotations

import logging

import keyring
from keyring.backend import KeyringBackend
from keyring.compat import properties
from keyring.errors import PasswordDeleteError

KEYRING_SERVICE = "astroquery:www.eso.org"

logger = logging.getLogger(__name__)


class EsoLoginError(RuntimeError):
    """The ESO archive did not accept the login."""


class SessionKeyring(KeyringBackend):
    """Keyring that keeps passwords in this process only, so nothing is written to disk."""

    @properties.classproperty
    def priority(cls) -> float:
        # Raising marks the backend as not viable, so keyring never selects it on its own.
        raise RuntimeError("SessionKeyring is installed explicitly by login_to_eso.")

    def __init__(self):
        super().__init__()
        self._passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self._passwords.pop((service, username), None) is None:
            raise PasswordDeleteError(f"No password for {username} in this session.")


def _stored_password(username: str) -> str | None:
    try:
        return keyring.get_password(KEYRING_SERVICE, username)
    except Exception:
        return None


def _use_memory_without_keyring(username: str) -> bool:
    """Install a :class:`SessionKeyring` when the system keyring cannot be read; True when one is active."""
    if isinstance(keyring.get_keyring(), SessionKeyring):
        return True
    try:
        keyring.get_password(KEYRING_SERVICE, username)
        return False
    except Exception as exc:
        logger.info("No usable keyring (%s); the ESO password is kept in memory until the process ends.", exc)
        keyring.set_keyring(SessionKeyring())
        return True


def login_to_eso(eso, username: str, store_password: bool) -> None:
    """Log ``eso`` in as ``username`` and check that ESO accepted it.

    With a working keyring this is ``eso.login`` as astroquery does it. Without one, the
    password asked for at the first login is kept in memory, so later logins of the process,
    including those of the observing-conditions step, do not ask again.

    Args:
        eso: An ``astroquery.eso.Eso`` instance.
        username: ESO user name.
        store_password: Save a newly entered password in the system keyring.

    Raises:
        EsoLoginError: ESO did not accept the login. A password entered for this login is
            forgotten, so the next login asks again.
    """
    in_memory = _use_memory_without_keyring(username)
    was_stored = _stored_password(username) is not None
    eso.login(username=username, store_password=store_password or in_memory)
    # astroquery's login wrapper drops the return value, so ask the session instead.
    if eso.authenticated():
        return
    if not was_stored:
        # astroquery saves the password before ESO has checked it.
        try:
            keyring.delete_password(KEYRING_SERVICE, username)
        except Exception:
            pass
    message = f"ESO did not accept the login as {username}: wrong password, or the ESO login service is down."
    if was_stored:
        message += f" If the password in the keyring is outdated, remove it with `keyring del {KEYRING_SERVICE} {username}`."
    raise EsoLoginError(message)

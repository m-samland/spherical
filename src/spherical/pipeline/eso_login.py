"""ESO archive login for one reduction run.

The download step opens a new ESO session and logs in for every observation, and the
observing-conditions step logs in only when the keyring already holds the password.
astroquery takes the password from the keyring and otherwise prompts, so on a machine
without a keychain service a batch would stop at every proprietary observation, and
proprietary SPARTA files would never be fetched. :func:`unlock_eso_account` asks once per
run, checks the password with ESO before anything is downloaded, and makes it available to
every later login of the process.
"""

from __future__ import annotations

import getpass
import logging

import keyring
from keyring.backend import KeyringBackend
from keyring.compat import properties

KEYRING_SERVICE = "astroquery:www.eso.org"

logger = logging.getLogger(__name__)


class EsoLoginError(RuntimeError):
    """The ESO archive refused the login."""


class SessionKeyring(KeyringBackend):
    """Keyring that keeps passwords in this process only.

    Reads fall back to the keyring it replaced, so passwords stored there stay usable.
    Nothing is written to disk, and the passwords are gone when the process ends.

    Args:
        fallback: The keyring that was active before; ``None`` for no fallback.
    """

    @properties.classproperty
    def priority(cls) -> float:
        # Raising marks the backend as not viable, so keyring never selects it on its own.
        raise RuntimeError("SessionKeyring is installed explicitly by unlock_eso_account.")

    def __init__(self, fallback: KeyringBackend | None = None):
        super().__init__()
        self._fallback = fallback
        self._passwords: dict[tuple[str, str], str] = {}

    def has_password(self, service: str, username: str) -> bool:
        return (service, username) in self._passwords

    def get_password(self, service: str, username: str) -> str | None:
        if (service, username) in self._passwords:
            return self._passwords[(service, username)]
        if self._fallback is None:
            return None
        try:
            return self._fallback.get_password(service, username)
        except Exception:
            return None

    def set_password(self, service: str, username: str, password: str) -> None:
        self._passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self._passwords.pop((service, username), None) is None and self._fallback is not None:
            self._fallback.delete_password(service, username)


def _keyring_password(username: str) -> str | None:
    try:
        return keyring.get_password(KEYRING_SERVICE, username)
    except Exception as exc:
        logger.info("No ESO password from the keyring (%s).", exc)
        return None


def _keep_in_memory(username: str, password: str) -> None:
    current = keyring.get_keyring()
    if not isinstance(current, SessionKeyring):
        current = SessionKeyring(fallback=current)
        keyring.set_keyring(current)
    current.set_password(KEYRING_SERVICE, username, password)


def unlock_eso_account(username: str, store_password: bool = False) -> None:
    """Make the ESO password of ``username`` available to every login of this process.

    A password already in the keyring is used as it is. Otherwise the password is asked
    for once. With ``store_password`` it is saved in the system keyring; when that is not
    possible, or ``store_password`` is False, it is kept in memory for the rest of the
    process. Either way the login is checked with ESO before returning.

    Args:
        username: ESO user name.
        store_password: Save a newly entered password in the system keyring.

    Raises:
        EsoLoginError: ESO refused the login.
    """
    from astroquery.eso import Eso

    current = keyring.get_keyring()
    if isinstance(current, SessionKeyring) and current.has_password(KEYRING_SERVICE, username):
        return
    from_keyring = _keyring_password(username) is not None
    if not from_keyring:
        password = getpass.getpass(f"ESO password for {username}: ")
        stored = False
        if store_password:
            try:
                keyring.set_password(KEYRING_SERVICE, username, password)
                stored = True
            except Exception as exc:
                logger.warning("Could not store the ESO password in the keyring (%s); keeping it in memory for this run.", exc)
        if not stored:
            _keep_in_memory(username, password)

    eso = Eso()
    eso.login(username=username, store_password=False)
    # astroquery's login wrapper drops the return value, so ask the session instead.
    if not eso.authenticated():
        if not from_keyring:
            try:
                keyring.delete_password(KEYRING_SERVICE, username)
            except Exception:
                pass
            raise EsoLoginError(f"The ESO archive refused the login as {username}. Check the user name and password.")
        raise EsoLoginError(
            f"The ESO archive refused the password stored in the keyring for {username}. "
            f"Remove it with `keyring del {KEYRING_SERVICE} {username}` and run again."
        )
    logger.info("ESO login as %s checked; later logins of this run reuse the password.", username)

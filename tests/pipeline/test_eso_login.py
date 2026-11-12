"""ESO login of the download step: one prompt per run without a keyring, astroquery's behaviour with one (#307)."""
from types import SimpleNamespace

import keyring
import pytest
from astroquery.eso import Eso
from astroquery.eso.core import EsoClass
from keyring.backend import KeyringBackend
from keyring.backends import fail
from keyring.compat import properties

from spherical.conditions.archive import _login_without_prompt
from spherical.pipeline.eso_login import KEYRING_SERVICE, EsoLoginError, SessionKeyring, login_to_eso

PASSWORD = "not-a-real-password"


class DictKeyring(KeyringBackend):
    """A working system keyring, kept in a dict."""

    @properties.classproperty
    def priority(cls) -> float:
        raise RuntimeError("test backend, installed explicitly")

    def __init__(self):
        super().__init__()
        self.passwords = {}

    def get_password(self, service, username):
        return self.passwords.get((service, username))

    def set_password(self, service, username, password):
        self.passwords[(service, username)] = password

    def delete_password(self, service, username):
        self.passwords.pop((service, username), None)


@pytest.fixture
def system_keyring():
    """Install a keyring for the test and restore the real one afterwards."""
    previous = keyring.get_keyring()
    yield lambda backend: keyring.set_keyring(backend) or backend
    keyring.set_keyring(previous)


@pytest.fixture
def prompts(monkeypatch):
    """Count password prompts; each answers PASSWORD."""
    asked = []

    def fake_getpass(prompt=""):
        asked.append(prompt)
        return PASSWORD

    monkeypatch.setattr("getpass.getpass", fake_getpass)
    return asked


@pytest.fixture
def eso(monkeypatch):
    """Replace only the network call of the ESO login, so astroquery's own login wrapper runs."""
    state = SimpleNamespace(accept=PASSWORD, logins=0)

    def fake_authenticate(self, *, username, password):
        state.logins += 1
        return password == state.accept

    monkeypatch.setattr(EsoClass, "_authenticate", fake_authenticate)
    return state


def _download_logins(count, store_password):
    """What the download step does for ``count`` observations that need files."""
    for _ in range(count):
        login_to_eso(Eso(), "user", store_password)


def test_without_a_keyring_the_password_is_asked_once(system_keyring, prompts, eso):
    system_keyring(fail.Keyring())

    _download_logins(3, store_password=False)

    assert len(prompts) == 1
    assert isinstance(keyring.get_keyring(), SessionKeyring)


def test_conditions_step_reuses_the_password_entered_at_the_download(system_keyring, prompts, eso):
    system_keyring(fail.Keyring())
    _download_logins(1, store_password=True)

    assert _login_without_prompt(Eso(), "user")
    assert len(prompts) == 1


def test_with_a_keyring_store_password_false_behaves_as_astroquery(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())

    _download_logins(2, store_password=False)

    assert len(prompts) == 2
    assert system.passwords == {}
    assert keyring.get_keyring() is system


def test_with_a_keyring_store_password_true_saves_it(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())

    _download_logins(2, store_password=True)

    assert len(prompts) == 1
    assert system.passwords == {(KEYRING_SERVICE, "user"): PASSWORD}
    assert keyring.get_keyring() is system


@pytest.mark.parametrize("backend", [DictKeyring, fail.Keyring])
def test_a_refused_new_password_raises_and_is_forgotten(system_keyring, prompts, eso, backend):
    system_keyring(backend())
    eso.accept = "something else"

    with pytest.raises(EsoLoginError, match="did not accept the login as user"):
        _download_logins(1, store_password=True)
    eso.accept = PASSWORD
    _download_logins(1, store_password=True)

    assert len(prompts) == 2


def test_a_refused_stored_password_is_kept_and_the_error_says_how_to_remove_it(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())
    system.set_password(KEYRING_SERVICE, "user", "stale")

    with pytest.raises(EsoLoginError, match="keyring del"):
        _download_logins(1, store_password=True)

    assert prompts == []
    assert system.passwords == {(KEYRING_SERVICE, "user"): "stale"}


def test_the_session_keyring_is_never_selected_on_its_own():
    assert SessionKeyring not in set(KeyringBackend.get_viable_backends())

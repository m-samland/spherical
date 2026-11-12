"""The ESO password is asked for once per run and reused by every later login (#307)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import keyring
import pytest
from astroquery.eso import Eso
from astroquery.eso.core import EsoClass
from keyring.backend import KeyringBackend
from keyring.backends import fail
from keyring.compat import properties

from spherical.pipeline import eso_login
from spherical.pipeline.eso_login import KEYRING_SERVICE, EsoLoginError, SessionKeyring, unlock_eso_account

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

    def install(backend):
        keyring.set_keyring(backend)
        return backend

    yield install
    keyring.set_keyring(previous)


@pytest.fixture
def prompts(monkeypatch):
    """Count password prompts; each answers PASSWORD."""
    asked = []

    def fake_getpass(prompt=""):
        asked.append(prompt)
        return PASSWORD

    monkeypatch.setattr(eso_login.getpass, "getpass", fake_getpass)
    monkeypatch.setattr("getpass.getpass", fake_getpass)
    return asked


@pytest.fixture
def eso(monkeypatch):
    """Replace only the network call of the ESO login; it accepts the password ``accept``.

    astroquery's own ``login`` wrapper, which reads the keyring and drops the return
    value, still runs.
    """
    state = SimpleNamespace(accept=PASSWORD, logins=0)

    def fake_authenticate(self, *, username, password):
        state.logins += 1
        return password == state.accept

    monkeypatch.setattr(EsoClass, "_authenticate", fake_authenticate)
    return state


def test_asks_once_without_a_keychain_and_later_logins_reuse_it(system_keyring, prompts, eso):
    system_keyring(fail.Keyring())

    unlock_eso_account("user")
    for _ in range(3):  # what the download of each later observation does
        session = Eso()
        session.login(username="user")
        assert session.authenticated()
    unlock_eso_account("user")

    assert len(prompts) == 1
    assert isinstance(keyring.get_keyring(), SessionKeyring)


def test_store_password_false_keeps_it_out_of_the_system_keyring(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())

    unlock_eso_account("user", store_password=False)

    assert system.passwords == {}
    assert keyring.get_password(KEYRING_SERVICE, "user") == PASSWORD


def test_store_password_true_saves_it_in_a_working_system_keyring(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())

    unlock_eso_account("user", store_password=True)

    assert system.passwords == {(KEYRING_SERVICE, "user"): PASSWORD}
    assert keyring.get_keyring() is system


def test_a_stored_password_is_used_without_asking(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())
    system.set_password(KEYRING_SERVICE, "user", PASSWORD)

    unlock_eso_account("user")

    assert prompts == []
    assert eso.logins == 1


def test_a_refused_new_password_raises_and_is_forgotten(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())
    eso.accept = "something else"

    with pytest.raises(EsoLoginError, match="refused the login as user"):
        unlock_eso_account("user", store_password=True)

    assert system.passwords == {}


def test_a_refused_stored_password_says_how_to_remove_it(system_keyring, prompts, eso):
    system = system_keyring(DictKeyring())
    system.set_password(KEYRING_SERVICE, "user", "stale")

    with pytest.raises(EsoLoginError, match="keyring del"):
        unlock_eso_account("user")


def test_the_session_keyring_is_never_selected_on_its_own():
    assert SessionKeyring not in set(KeyringBackend.get_viable_backends())


class TestExecuteTargets:
    @pytest.fixture
    def ir(self, monkeypatch):
        pytest.importorskip("charis")
        from spherical.pipeline import ifs_reduction as ir

        monkeypatch.setattr(ir, "execute_target", MagicMock())
        return ir

    @staticmethod
    def config(**preprocessing):
        from spherical.pipeline.pipeline_config import defaultIFSReduction

        config = defaultIFSReduction()
        config.preprocessing = config.preprocessing.merge(**preprocessing)
        return config

    observations = [SimpleNamespace(observation={"INSTRUMENT": ["IFS"]}) for _ in range(3)]

    def test_unlocks_once_before_the_first_observation(self, ir, monkeypatch):
        calls = []
        monkeypatch.setattr(ir, "unlock_eso_account", lambda *a, **k: calls.append(ir.execute_target.call_count))
        config = self.config(eso_username="user", delete_password_after_reduction=False)

        ir.execute_targets(observations=self.observations, config=config)

        assert calls == [0]
        assert ir.execute_target.call_count == 3

    def test_a_refused_login_stops_before_any_observation(self, ir, monkeypatch):
        def refuse(*args, **kwargs):
            raise EsoLoginError("refused")

        monkeypatch.setattr(ir, "unlock_eso_account", refuse)

        with pytest.raises(EsoLoginError):
            ir.execute_targets(observations=self.observations, config=self.config(eso_username="user"))
        ir.execute_target.assert_not_called()

    def test_no_username_no_login(self, ir, monkeypatch):
        unlock = MagicMock()
        monkeypatch.setattr(ir, "unlock_eso_account", unlock)

        ir.execute_targets(observations=self.observations, config=self.config())

        unlock.assert_not_called()

    def test_no_login_when_no_step_uses_the_archive(self, ir, monkeypatch):
        unlock = MagicMock()
        monkeypatch.setattr(ir, "unlock_eso_account", unlock)
        config = self.config(eso_username="user", delete_password_after_reduction=False)
        config.steps = config.steps.merge(download_data=False, observing_conditions=False)

        ir.execute_targets(observations=self.observations, config=config)

        unlock.assert_not_called()

"""run_trap_on_observations with one TRAP config per observation."""
from types import SimpleNamespace

import pytest

# run_trap imports trap at module level.
pytest.importorskip("trap")

from spherical.pipeline import run_trap  # noqa: E402
from spherical.pipeline.pipeline_config import IFSReductionConfig  # noqa: E402


def _config():
    return SimpleNamespace(processing=SimpleNamespace(verbose=False))


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_run(observation, trap_config, reduction_config, species_database_directory):
        calls.append((observation, trap_config))

    monkeypatch.setattr(run_trap, "run_trap_on_observation", fake_run)
    return calls


def test_callable_config_resolved_per_observation(recorded):
    configs = {"a": _config(), "b": _config()}
    run_trap.run_trap_on_observations(["a", "b"], configs.__getitem__, IFSReductionConfig(), "species")
    assert recorded == [("a", configs["a"]), ("b", configs["b"])]


def test_plain_config_passed_to_every_observation(recorded):
    config = _config()
    run_trap.run_trap_on_observations(["a", "b"], config, IFSReductionConfig(), "species")
    assert recorded == [("a", config), ("b", config)]


def test_callable_failure_skips_only_that_observation(recorded):
    config = _config()

    def lookup(observation):
        if observation == "a":
            raise KeyError(observation)
        return config

    run_trap.run_trap_on_observations(["a", "b"], lookup, IFSReductionConfig(), "species")
    assert recorded == [("b", config)]

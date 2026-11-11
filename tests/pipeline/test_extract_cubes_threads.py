"""Thread budget of the IFS extraction pool (#241)."""

import os

import pandas as pd
import pytest
from astropy.io import fits

pytest.importorskip("charis")

from spherical.pipeline.steps import extract_cubes as ec  # noqa: E402


class _RecordingPool:
    calls: list = []

    def __init__(self, **kwargs):
        # The limit is set in the parent, so it must be active when the pool starts.
        self.kwargs = {**kwargs, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap(self, func, iterable):
        tasks = [task for task, _ in iterable]
        _RecordingPool.calls.append((self.kwargs, tasks))
        return [True] * len(tasks)


def _extract(tmp_path, monkeypatch, ndit, ncpu, maxcpus=1):
    raw = tmp_path / "flux.fits"
    hdu = fits.PrimaryHDU()
    hdu.header["HIERARCH ESO DET NDIT"] = ndit
    hdu.writeto(raw)

    _RecordingPool.calls = []
    monkeypatch.setattr(ec, "WORKER_CONTEXT", type("Context", (), {"Pool": _RecordingPool}))
    monkeypatch.setattr(ec, "cpu_count", lambda: 64)
    observation = type("Obs", (), {"frames": {"FLUX": pd.DataFrame({"FILE": [str(raw)], "MJD_OBS": [57000.0]})}})()
    ec.extract_cubes_with_multiprocessing(
        observation,
        ["FLUX"],
        {"bgsub": True, "fitbkgnd": False, "fitshift": True, "maxcpus": maxcpus},
        {"ncpu_cubebuilding": ncpu, "bg_pca": True},
        str(tmp_path / "wavecal"),
        str(tmp_path / "cubes"),
    )
    [(kwargs, tasks)] = _RecordingPool.calls
    return kwargs, tasks


def test_workers_are_limited_to_one_thread_when_dits_fill_the_cpus(tmp_path, monkeypatch):
    kwargs, tasks = _extract(tmp_path, monkeypatch, ndit=8, ncpu=4)
    assert kwargs == {"processes": 4, "OMP_NUM_THREADS": "1"}
    assert {task[-1]["maxcpus"] for task in tasks} == {1}


def test_spare_cpus_go_to_charis_and_blas_threads(tmp_path, monkeypatch):
    kwargs, tasks = _extract(tmp_path, monkeypatch, ndit=2, ncpu=8)
    assert kwargs == {"processes": 2, "OMP_NUM_THREADS": "4"}
    assert {task[-1]["maxcpus"] for task in tasks} == {4}


def test_explicit_maxcpus_is_not_lowered(tmp_path, monkeypatch):
    _, tasks = _extract(tmp_path, monkeypatch, ndit=8, ncpu=4, maxcpus=3)
    assert {task[-1]["maxcpus"] for task in tasks} == {3}

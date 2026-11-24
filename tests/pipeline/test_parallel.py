"""Thread limits (#241) and worker failures (#289) in spherical's process pools."""

import os
import subprocess
import sys
import threading
import time

import pytest
from threadpoolctl import threadpool_info

from spherical.pipeline import parallel
from spherical.pipeline.parallel import THREAD_ENV_VARS, limit_worker_threads, parallel_map_ordered, worker_thread_budget


def _max_threads(index):
    # Imported here, after the initializer: under spawn this is the first BLAS load in the worker.
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401

    return index, max(info["num_threads"] for info in threadpool_info())


@pytest.fixture
def unlimited_env(monkeypatch):
    """Remove the shell's thread limits so the workers start unlimited, as on a clean node."""
    for var in THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "ncpu, ntasks, expected",
    [
        (10, 100, (10, 1)),
        (10, 10, (10, 1)),
        (10, 3, (3, 3)),
        (10, 1, (1, 10)),
        (4, 0, (1, 4)),
        (1, 5, (1, 1)),
    ],
)
def test_worker_thread_budget_never_exceeds_ncpu(ncpu, ntasks, expected):
    nproc, nthreads = worker_thread_budget(ncpu, ntasks)
    assert (nproc, nthreads) == expected
    assert nproc * nthreads <= ncpu


@pytest.mark.usefixtures("unlimited_env")
def test_parallel_map_ordered_limits_threads_per_worker():
    results = parallel_map_ordered(_max_threads, list(range(4)), ncpu=2)
    assert results == [1, 1, 1, 1]


@pytest.mark.usefixtures("unlimited_env")
def test_parallel_map_ordered_splits_spare_cores_between_workers():
    results = parallel_map_ordered(_max_threads, list(range(2)), ncpu=4)
    assert results == [2, 2]


@pytest.mark.usefixtures("unlimited_env")
def test_limit_restores_the_parent_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with limit_worker_threads(2):
        assert {os.environ[var] for var in THREAD_ENV_VARS} == {"2"}
    assert os.environ["OMP_NUM_THREADS"] == "7"
    assert all(var not in os.environ for var in THREAD_ENV_VARS if var != "OMP_NUM_THREADS")


@pytest.mark.usefixtures("unlimited_env")
def test_workers_keep_the_limit_with_numba_loaded():
    # numba starts the OpenMP runtime at import. Forked workers inherited it, and on
    # Linux with conda-forge's OpenMP OpenBLAS they then aborted (IRDIS preprocessing
    # on the MPCDF node); spawned workers start clean.
    pytest.importorskip("numba")
    assert parallel_map_ordered(_max_threads, list(range(4)), ncpu=2) == [1, 1, 1, 1]


def test_pools_spawn_their_workers():
    assert parallel.WORKER_CONTEXT.get_start_method() == "spawn"


def _fail_first_task(task):
    index, _ = task
    time.sleep(0.5)
    if index == 0:
        raise ValueError("task 0 failed")
    return index, index


def test_worker_exception_stops_the_map_instead_of_hanging():
    # While the only worker spends 0.5 s on the first task, the pool's task handler is blocked
    # sending the second, which is larger than the pipe buffer. Pool.terminate() drains such
    # a task only while the pipe holds data, so whether it frees the handler depends on timing
    # (#289). The drain is disabled here so that the race is lost every time. A hang would
    # block pytest, hence the subprocess and its timeout.
    code = f"""
import sys
sys.path.insert(0, {os.path.dirname(__file__)!r})
from multiprocessing.pool import Pool
import numpy as np
from spherical.pipeline.parallel import parallel_map_ordered
from test_parallel import _fail_first_task

Pool._help_stuff_finish = staticmethod(lambda *args: None)

if __name__ == "__main__":
    try:
        parallel_map_ordered(_fail_first_task, [(i, np.zeros(2**20)) for i in range(4)], ncpu=1)
    except ValueError as exc:
        print("raised:", exc)
"""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("parallel_map_ordered hung after a worker raised")
    assert "raised: task 0 failed" in result.stdout, result.stderr


class _TwoArgumentError(Exception):
    """Pickles, but cannot be rebuilt from its args, which kills the pool's result handler."""

    def __init__(self, what, why):
        super().__init__(f"{what}: {why}")


def _fail_without_pickle_round_trip(index):
    if index == 0:
        raise _TwoArgumentError("task 0", "failed")
    return index, index


def _fail_unpicklable(index):
    if index == 0:
        error = ValueError("task 0 failed")
        error.lock = threading.Lock()
        raise error
    return index, index


@pytest.mark.parametrize("func", ["_fail_without_pickle_round_trip", "_fail_unpicklable"])
def test_worker_exception_that_does_not_pickle_still_stops_the_map(func):
    code = f"""
import sys
sys.path.insert(0, {os.path.dirname(__file__)!r})
from spherical.pipeline.parallel import parallel_map_ordered
from test_parallel import {func}

if __name__ == "__main__":
    try:
        parallel_map_ordered({func}, list(range(4)), ncpu=1)
    except RuntimeError as exc:
        print("raised:", exc)
"""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("parallel_map_ordered hung after a worker raised")
    assert "raised:" in result.stdout and "task 0" in result.stdout, result.stderr


def test_worker_exception_keeps_the_worker_traceback():
    with pytest.raises(ValueError, match="task 0 failed") as excinfo:
        parallel_map_ordered(_fail_first_task, [(i, None) for i in range(3)], ncpu=2)
    assert "_fail_first_task" in str(excinfo.value.__cause__)

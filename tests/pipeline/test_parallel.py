"""Thread limits inside spherical's process pools (#241)."""

import pytest
from threadpoolctl import threadpool_info

from spherical.pipeline.parallel import THREAD_ENV_VARS, parallel_map_ordered, worker_thread_budget


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

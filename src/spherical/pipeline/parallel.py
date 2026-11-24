import multiprocessing
import os
import pickle
import traceback
from contextlib import contextmanager
from multiprocessing.pool import RemoteTraceback
from typing import Any, Callable, List, NamedTuple, Tuple

from tqdm.auto import tqdm

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: Start method of every spherical process pool. A forked worker inherits the
#: parent's OpenMP runtime, and GNU OpenMP (conda-forge's OpenBLAS on Linux) is
#: not usable in a child once the parent has started it, which numba does at
#: import: BLAS or threadpoolctl calls in the worker then abort or hang.
WORKER_CONTEXT = multiprocessing.get_context("spawn")


def worker_thread_budget(ncpu: int, ntasks: int) -> Tuple[int, int]:
    """Split ``ncpu`` cores into worker processes and threads per worker.

    One process per task up to ``ncpu``; cores left over when there are fewer
    tasks than cores go to the threads of each worker, so that
    ``processes * threads <= ncpu``.

    Returns:
        ``(processes, threads_per_worker)``, both at least 1.
    """
    processes = max(1, min(ncpu, ntasks))
    return processes, max(1, ncpu // processes)


@contextmanager
def limit_worker_threads(nthreads: int):
    """Cap the BLAS and OpenMP threads of the pool workers started inside the block.

    Without it every worker starts one thread per core, so a pool of ``ncpu``
    workers oversubscribes the machine ``ncpu`` times over. The workers are
    spawned (:data:`WORKER_CONTEXT`), so their libraries read the limit from the
    environment variables set here when they load. The parent's own libraries
    are not touched.
    """
    saved = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    os.environ.update({var: str(nthreads) for var in THREAD_ENV_VARS})
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def parallel_map_ordered(
    func: Callable[[Any], Tuple[int, Any]],
    args_list: List[Any],
    ncpu: int = 4,
    desc: str = "Processing"
) -> List[Any]:
    """
    Multiprocessing helper that applies `func` to elements in `args_list`.

    - `func(arg)` must return (index, result)
    - Uses unordered mapping for performance
    - Restores original order by index
    - Displays tqdm progress bar
    - Splits `ncpu` between worker processes and their BLAS/OpenMP threads

    Returns: List of results in the same order as `args_list`

    Raises: The first exception to arrive from a worker, once every task has run.
    """
    processes, nthreads = worker_thread_budget(ncpu, len(args_list))
    results = []
    failure = None
    with limit_worker_threads(nthreads), WORKER_CONTEXT.Pool(processes=processes) as pool:
        for result in tqdm(pool.imap_unordered(_ReturnFailure(func), args_list), total=len(args_list), desc=desc):
            if isinstance(result, _WorkerFailure):
                failure = failure or result
            else:
                results.append(result)

    if failure is not None:
        raise failure.exception from RemoteTraceback(failure.traceback)
    results.sort(key=lambda x: x[0])
    return [res[1] for res in results]


class _WorkerFailure(NamedTuple):
    exception: BaseException
    traceback: str


class _ReturnFailure:
    """Pool task that returns the exception of ``func`` instead of raising it.

    An exception raised in a worker reaches ``imap_unordered`` in the parent, and
    leaving the pool on it calls ``Pool.terminate()`` while tasks may still be
    queued. ``terminate()`` then joins the task-handler thread, which hangs for
    good if that thread is blocked sending a task larger than the pipe buffer
    (#289). Returning the exception lets every task be sent and the pool close
    normally.
    """

    def __init__(self, func: Callable[[Any], Tuple[int, Any]]):
        self.func = func

    def __call__(self, arg: Any):
        try:
            return self.func(arg)
        except Exception as exc:
            worker_traceback = traceback.format_exc()
            # An exception that cannot make the round trip to the parent would
            # stop the pool's result handler or be raised inside the pool, which
            # brings back the hang.
            try:
                pickle.loads(pickle.dumps(exc))
            except Exception:
                exc = RuntimeError(f"{type(exc).__name__}: {exc}")
            return _WorkerFailure(exc, worker_traceback)

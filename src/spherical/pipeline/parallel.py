import multiprocessing
import os
from contextlib import contextmanager
from typing import Any, Callable, List, Tuple

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
    """
    processes, nthreads = worker_thread_budget(ncpu, len(args_list))
    with limit_worker_threads(nthreads), WORKER_CONTEXT.Pool(processes=processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(func, args_list), total=len(args_list), desc=desc):
            results.append(result)

    results.sort(key=lambda x: x[0])
    return [res[1] for res in results]

import os
from contextlib import contextmanager
from multiprocessing import Pool
from typing import Any, Callable, List, Tuple

from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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
    workers oversubscribes the machine ``ncpu`` times over. The limit is set in
    the parent before the pool starts: forked workers inherit the limit of the
    libraries already loaded, spawned workers the environment variables. The
    workers do not call ``threadpoolctl`` themselves, because doing so in a
    freshly forked worker can crash it on Linux once numba is loaded.
    """
    saved = {var: os.environ.get(var) for var in THREAD_ENV_VARS}
    os.environ.update({var: str(nthreads) for var in THREAD_ENV_VARS})
    try:
        with threadpool_limits(nthreads):
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
    with limit_worker_threads(nthreads), Pool(processes=processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(func, args_list), total=len(args_list), desc=desc):
            results.append(result)

    results.sort(key=lambda x: x[0])
    return [res[1] for res in results]

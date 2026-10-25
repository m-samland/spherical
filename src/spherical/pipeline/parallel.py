import os
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


def limit_worker_threads(nthreads: int) -> None:
    """Pool initializer that caps the BLAS and OpenMP threads of a worker process.

    Without it every worker starts one thread per core, so a pool of ``ncpu``
    workers oversubscribes the machine ``ncpu`` times over. The environment
    variables cover libraries the worker loads later; ``threadpool_limits``
    covers the ones already loaded, which is all of them under ``fork``.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(nthreads)
    threadpool_limits(nthreads)


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
    with Pool(processes=processes, initializer=limit_worker_threads, initargs=(nthreads,)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(func, args_list), total=len(args_list), desc=desc):
            results.append(result)

    results.sort(key=lambda x: x[0])
    return [res[1] for res in results]

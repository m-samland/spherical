from multiprocessing import Pool
from typing import Any, Callable, List, Tuple

from tqdm import tqdm


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

    Returns: List of results in the same order as `args_list`
    """
    with Pool(processes=ncpu) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(func, args_list), total=len(args_list), desc=desc):
            results.append(result)

    results.sort(key=lambda x: x[0])
    return [res[1] for res in results]

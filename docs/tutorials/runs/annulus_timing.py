"""Time TRAP on an annulus around 51 Eri b, next to the tutorial's full-field run.

TRAP writes its results into the reduction directory without a suffix, so an annulus
run in the tutorial's own folder would replace the full-field results the tutorial
shows. This script gives the annulus run its own folder, `<source label>_annulus`,
whose observation tree is hard-linked to the tutorial run (no extra disk space), and
keeps the shared files read-only while TRAP runs.

    python docs/tutorials/runs/annulus_timing.py --instrument irdis --ncpu 8 --inner 31 --outer 43
    python docs/tutorials/runs/annulus_timing.py --instrument ifs --ncpu 32 --inner 50 --outer 72

The bounds are the ones of the 51 Eri regression runs in tests/regression/.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import stat
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from tutorial_helpers import tutorial_dir  # noqa: E402


def hardlink_tree(src: Path, dst: Path) -> None:
    """Recreate the folders of `src` under `dst` and hard-link every file."""
    for root, _dirs, files in os.walk(src):
        target = dst / Path(root).relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            os.link(Path(root) / name, target / name)


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file under `root`."""
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            info = path.stat()
            result[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns)
    return result


def changed_files(before: dict, after: dict) -> list[str]:
    return sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))


@contextlib.contextmanager
def read_only(root: Path):
    """Remove write permission from the files under `root` and restore it afterwards."""
    modes = {path: path.stat().st_mode for path in root.rglob("*") if path.is_file()}
    try:
        for path, mode in modes.items():
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        yield
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instrument", choices=["ifs", "irdis"], required=True)
    parser.add_argument("--ncpu", type=int, required=True)
    parser.add_argument("--inner", type=int, required=True, help="search_region_inner_bound, px")
    parser.add_argument("--outer", type=int, required=True, help="search_region_outer_bound, px")
    parser.add_argument("--species-dir", type=Path, default=Path.home() / "data/sphere/species")
    args = parser.parse_args(argv)

    from spherical.pipeline.run_trap import run_trap_on_observations

    run = importlib.import_module(f"51eri_{args.instrument}")
    source, label = run.LABEL, f"{run.LABEL}_annulus"
    inst = args.instrument.upper()
    src_obs = tutorial_dir() / source / "reduction" / inst / "observation"
    dst_obs = tutorial_dir() / label / "reduction" / inst / "observation"
    if not src_obs.is_dir():
        print(f"{src_obs} does not exist; run docs/tutorials/runs/51eri_{args.instrument}.py first.")
        return 1
    if dst_obs.exists():
        print(f"{dst_obs.parent.parent} exists from an earlier annulus run; remove it first.")
        return 1

    config, trap_config = run.build(args.ncpu, label)
    trap_config.reduction = trap_config.reduction.merge(
        search_region_inner_bound=args.inner, search_region_outer_bound=args.outer)
    database_dir = tutorial_dir() / "database" if (tutorial_dir() / "database").is_dir() else None
    table, observations = run.template.select_observations(
        [run.runner.TARGET], database_directory=database_dir, OBS_ID=run.OBS_ID)
    hardlink_tree(src_obs, dst_obs)

    before = snapshot(src_obs)
    with read_only(src_obs):
        start = time.monotonic()
        run_trap_on_observations(observations=observations, trap_config=trap_config,
                                 reduction_config=config, species_database_directory=args.species_dir)
        seconds = time.monotonic() - start
    changed = changed_files(before, snapshot(src_obs))
    if changed:
        print(f"TRAP changed files of the tutorial run: {changed}")
        return 1

    result = {"instrument": args.instrument, "inner": args.inner, "outer": args.outer,
              "ncpu": args.ncpu, "seconds": round(seconds, 1), "source": source}
    (tutorial_dir() / label / "annulus_timing.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"TRAP on the {args.inner} to {args.outer} px annulus took {seconds / 60:.1f} min "
          f"with set_ncpu({args.ncpu}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared part of the tutorial run scripts.

A run script reduces one sequence for a tutorial with the settings of a reduction
template, changing only the CPU count and the directories. It writes
`run_provenance.json` into `$SPHERICAL_TUTORIAL_DIR/<label>/` before it starts, so the
tutorial's stamp names the code and machine that produced the products.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for folder in ("examples", "docs/tools", "docs/tutorials"):
    if str(REPO / folder) not in sys.path:
        sys.path.insert(0, str(REPO / folder))

import stamp_notebook  # noqa: E402
from tutorial_helpers import tutorial_dir  # noqa: E402

from spherical.pipeline.ifs_reduction import execute_targets  # noqa: E402
from spherical.pipeline.run_trap import run_trap_on_observations  # noqa: E402

TARGET = "51 Eri"


def build(template, ncpu: int, label: str):
    """The template's configuration with the tutorial directories."""
    config = template.build_config(ncpu=ncpu, base_path=tutorial_dir())
    config.directories.raw_directory = tutorial_dir() / "data"
    config.directories.reduction_directory = tutorial_dir() / label / "reduction"
    return config, template.build_trap_config(config)


def source_hash(template, build_function) -> str:
    """Hash of the code that sets the configuration, to notice later template edits."""
    sources = [inspect.getsource(f) for f in (template.build_config, template.build_trap_config, build_function)]
    return hashlib.sha256("".join(sources).encode()).hexdigest()


def main(template, build_function, hash_function, label: str, obs_id: int, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Reduce {TARGET}, OBS_ID {obs_id}, for the tutorial.")
    parser.add_argument("--ncpu", type=int, required=True, help="value for config.set_ncpu")
    parser.add_argument("--label", default=label, help="run folder name; a timing run uses its own")
    parser.add_argument("--no-trap", action="store_true", help="stop after the preprocessing (timing runs)")
    parser.add_argument("--species-dir", type=Path, default=Path.home() / "data/sphere/species",
                        help="species database for TRAP's template matching")
    parser.add_argument("--database-dir", type=Path, default=None,
                        help="observation tables; default $SPHERICAL_TUTORIAL_DIR/database if it exists")
    parser.add_argument("--resume", action="store_true", help="continue a run whose folder is not empty")
    args = parser.parse_args(argv)

    run_dir = tutorial_dir() / args.label
    reduction = run_dir / "reduction"
    if reduction.is_dir() and any(reduction.iterdir()) and not args.resume:
        print(f"{reduction} is not empty. Timings need a fresh run: remove it, choose another "
              "--label, or pass --resume.")
        return 1
    database_dir = args.database_dir
    if database_dir is None and (tutorial_dir() / "database").is_dir():
        database_dir = tutorial_dir() / "database"

    config, trap_config = build_function(args.ncpu, args.label)
    table, observations = template.select_observations([TARGET], database_directory=database_dir, OBS_ID=obs_id)
    if len(table) != 1:
        print(f"Expected one observation of {TARGET} with OBS_ID {obs_id}, found {len(table)}.")
        return 1
    stamp_notebook.write_run_provenance(
        run_dir / stamp_notebook.RUN_PROVENANCE, ncpu=args.ncpu,
        extra={"label": args.label, "obs_id": obs_id, "no_trap": args.no_trap, "source_hash": hash_function()})

    execute_targets(observations=observations, config=config)
    if not args.no_trap:
        run_trap_on_observations(observations=observations, trap_config=trap_config,
                                 reduction_config=config, species_database_directory=args.species_dir)
    return 0

"""Measure run time per step and disk use per stage of a tutorial reduction.

Reads the JSON logs of one run in `$SPHERICAL_TUTORIAL_DIR/<label>/` and writes
`docs/_data/requirements_51eri_steps.csv` and `docs/_data/requirements_51eri_disk.csv`,
which the Practical requirements page renders. Standard library only, so it also runs
on a machine without pixi, where the data are.

    python docs/tools/measure_requirements.py --instrument irdis --label 51eri_irdis \\
        --target "*_51_Eri" --filter DB_K12 --date 2015-09-24
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "tutorials"))
from tutorial_helpers import tutorial_dir  # noqa: E402

TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
CLOSING = {"success", "skipped", "skipped_complete"}
SESSION_MARKERS = {"session_start", "trap_session_start"}
IGNORED_STEPS = {"trap_session"}
FRAME_TYPES = ("coro", "center", "flux")
STEPS_FIELDS = ["instrument", "label", "ncpu", "frames_coro", "frames_center", "frames_flux", "step", "depth", "seconds"]
DISK_FIELDS = ["instrument", "label", "stage", "bytes", "gb"]


@dataclass
class StepTime:
    step: str
    start: datetime
    end: datetime | None
    depth: int = 0

    @property
    def seconds(self) -> float | None:
        return None if self.end is None else (self.end - self.start).total_seconds()


def _rotation_index(path: Path) -> int:
    """`reduction.jsonlog.3` is the oldest file, the bare `reduction.jsonlog` the newest."""
    suffix = path.name.rsplit(".", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _records(paths: list[Path]) -> list[dict]:
    records = []
    for path in sorted(paths, key=_rotation_index, reverse=True):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("step") and record.get("asctime"):
                records.append(record)
    return sorted(records, key=lambda r: r["asctime"])  # stable: equal stamps keep file order


def read_step_times(paths: list[Path]) -> tuple[list[StepTime], int]:
    """Pair each step's `started` with its closing record; return steps and session count.

    A repeated `started` of an open step is ignored (extract_cubes logs two), `failed`
    inside a step is a warning rather than an end, and a step never closed is kept with
    `end=None`.
    """
    steps: list[StepTime] = []
    open_steps: dict[str, StepTime] = {}
    sessions = 0
    for record in _records(paths):
        name, status = record["step"], record.get("status")
        when = datetime.strptime(record["asctime"], TIME_FORMAT)
        if name in SESSION_MARKERS:
            sessions += status == "started"
        elif name in IGNORED_STEPS:
            continue
        elif status == "started" and name not in open_steps:
            open_steps[name] = StepTime(name, when, None)
            steps.append(open_steps[name])
        elif status in CLOSING and name in open_steps:
            open_steps.pop(name).end = when
    for step in steps:
        inside = any(other is not step and other.end is not None and step.end is not None
                     and other.start <= step.start and step.end <= other.end for other in steps)
        step.depth = int(inside)
    return steps, sessions


def directory_bytes(path: Path) -> int:
    """Size of the regular files under `path`; symbolic links are not followed."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file = Path(root) / name
            if not file.is_symlink():
                total += file.stat().st_size
    return total


def stage_paths(instrument: str, raw_dir: Path, reduction_dir: Path, target: str, filt: str, date: str) -> dict[str, list[Path]]:
    """Folders of each stage, following the layout on the Output products page."""
    inst = instrument.upper()
    observation = reduction_dir / inst / "observation" / target / filt / date
    stages = {
        "raw_science": [raw_dir / inst / "science" / target / filt / date],
        "raw_calibration": [raw_dir / inst / "calibration" / filt],
        "trap": [reduction_dir / inst / "trap" / target / filt / date],
    }
    if inst == "IFS":
        stages["calibration"] = [reduction_dir / inst / "calibration" / filt]
        stages["extracted"] = [observation / "optext" / kind.upper() for kind in FRAME_TYPES]
        stages["products"] = [observation / "optext" / "converted"]
    else:
        stages["calibration"] = [reduction_dir / inst / "calibration" / filt / date]
        stages["products"] = [observation / "converted"]
    return stages


def frame_counts(products: Path) -> dict[str, int]:
    """Frames per type, from the data rows of `frames_info_{type}.csv`."""
    counts = {}
    for kind in FRAME_TYPES:
        path = products / f"frames_info_{kind}.csv"
        rows = 0
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
        counts[f"frames_{kind}"] = rows
    return counts


def write_rows(csv_path: Path, key: tuple[str, str], rows: list[dict]) -> None:
    """Replace the rows of one (instrument, label) and keep all others."""
    kept, fields = [], ["instrument", "label"]
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or fields)
            kept = [row for row in reader if (row["instrument"], row["label"]) != key]
    new = [{"instrument": key[0], "label": key[1], **row} for row in rows]
    for row in new:
        fields += [name for name in row if name not in fields]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept + new)


def _with_rotations(base: Path) -> list[Path]:
    candidates = [base] + [base.with_name(f"{base.name}.{i}") for i in (1, 2, 3)]
    return [path for path in candidates if path.exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instrument", choices=["ifs", "irdis"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--filter", required=True, dest="filt")
    parser.add_argument("--date", required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="accept logs of a resumed or repeated run")
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs" / "_data")
    args = parser.parse_args(argv)

    run_dir = tutorial_dir() / args.label
    raw_dir, reduction_dir = tutorial_dir() / "data", run_dir / "reduction"
    stages = stage_paths(args.instrument, raw_dir, reduction_dir, args.target, args.filt, args.date)
    observation = reduction_dir / args.instrument.upper() / "observation" / args.target / args.filt / args.date

    problems = []
    reduction_steps, sessions = read_step_times(_with_rotations(observation / "reduction.jsonlog"))
    if sessions != 1:
        problems.append(f"reduction log holds {sessions} sessions")
    trap_steps, trap_sessions = read_step_times(_with_rotations(stages["trap"][0] / "trap_reduction.jsonlog"))
    if trap_sessions > 1:
        problems.append(f"TRAP log holds {trap_sessions} sessions")
    for folder in (observation / "old_logs", stages["trap"][0] / "old_logs"):
        if folder.is_dir() and any(folder.iterdir()):
            problems.append(f"{folder} is not empty (an earlier run's logs)")
    if problems and not args.allow_partial:
        print("Timings would be partial: " + "; ".join(problems) + ". Use --allow-partial to accept.")
        return 1

    provenance = run_dir / "run_provenance.json"
    ncpu = json.loads(provenance.read_text())["ncpu"] if provenance.exists() else ""
    frames = frame_counts(stages["products"][0])
    step_rows = [{"ncpu": ncpu, **frames, "step": step.step, "depth": step.depth,
                  "seconds": "" if step.seconds is None else round(step.seconds, 1)}
                 for step in reduction_steps]
    step_rows += [{"ncpu": ncpu, **frames, "step": f"trap:{step.step}", "depth": step.depth,
                   "seconds": "" if step.seconds is None else round(step.seconds, 1)}
                  for step in trap_steps]
    disk_rows = []
    for stage, folders in stages.items():
        size = sum(directory_bytes(folder) for folder in folders)
        disk_rows.append({"stage": stage, "bytes": size, "gb": round(size / 1e9, 2)})

    key = (args.instrument, args.label)
    write_rows(args.out_dir / "requirements_51eri_steps.csv", key, step_rows)
    write_rows(args.out_dir / "requirements_51eri_disk.csv", key, disk_rows)
    for row in step_rows:
        print(f"{'  ' * row['depth']}{row['step']:<40} {row['seconds']:>10} s")
    for row in disk_rows:
        print(f"{row['stage']:<20} {row['gb']:>8} GB")
    print(f"Frames: {frames}. Wrote {args.out_dir}/requirements_51eri_{{steps,disk}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
aggregate_reduction_status.py

Summarise SPHERE reduction status across many JSON logs.

Covers both instruments (IFS and IRDIS) and both pipelines (the reduction
itself and the TRAP post-processing), which write ``reduction.jsonlog`` and
``trap_reduction.jsonlog`` respectively.

Usage
-----
    python aggregate_reduction_status.py /path/to/reductions [--csv summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from spherical.scripts._monitoring import format_table, instrument_from_band

# Step whose success means the pipeline ran to the end. IRDIS shares the IFS
# entry: its last pre-TRAP step is `spot_to_flux`, which reuses the IFS StepSpec
# and so logs under the same name.
FINAL_STEPS = {
    "reduction": "spot_to_flux_normalization",
    "trap": "trap_session",
}

LOG_PATTERNS = {
    "reduction": "reduction.jsonlog",
    "trap": "trap_reduction.jsonlog",
}


def detect_pipeline(jsonlog_path: Path) -> str:
    """Return which pipeline wrote this log, from its file name."""
    return "trap" if "trap" in jsonlog_path.name.lower() else "reduction"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise SPHERE pipeline reductions.")
    p.add_argument(
        "root_dir",
        type=Path,
        help="Directory that contains many sub-folders with log files",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the summary table as CSV",
    )
    p.add_argument(
        "--pipeline",
        choices=["reduction", "trap", "all"],
        default="all",
        help="Filter by pipeline (default: all)",
    )
    p.add_argument(
        "--instrument",
        choices=["ifs", "irdis", "all"],
        default="all",
        help="Filter by instrument (default: all)",
    )
    return p.parse_args()


def extract_structured_rows(jsonlog: Path) -> list[dict]:
    """Return rows that contain all required structured fields."""
    rows: list[dict] = []
    with jsonlog.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # keep only records that have a non-null structured payload
            if all(rec.get(k) is not None for k in ("target", "band", "night", "step", "status")):
                rows.append(
                    {
                        "target": rec["target"],
                        "band": rec["band"],
                        "night": rec["night"],
                        "step": rec["step"],
                        "status": rec["status"],
                    }
                )
    return rows


def aggregate(root: Path) -> list[dict]:
    """
    Return a list of dicts:
    {
        'target': str,
        'instrument': str,
        'band': str,
        'night': str,
        'pipeline': str,
        'complete': bool,
        'last_step': str,
        'last_status': str,
        'log_path': str
    }
    """
    summary: dict[tuple, dict] = {}

    for pattern in LOG_PATTERNS.values():
        for jsonlog in root.rglob(pattern):
            pipeline = detect_pipeline(jsonlog)
            final_step = FINAL_STEPS[pipeline]

            rows = extract_structured_rows(jsonlog)
            for r in rows:
                key = (r["target"], r["band"], r["night"], pipeline)
                current = summary.get(
                    key,
                    {
                        "target": r["target"],
                        "instrument": instrument_from_band(r["band"]),
                        "band": r["band"],
                        "night": r["night"],
                        "pipeline": pipeline,
                        "complete": False,
                        "last_step": None,
                        "last_status": None,
                        "log_path": str(jsonlog),
                    },
                )

                # Always update "last_*" because logs are naturally chronological
                current["last_step"] = r["step"]
                current["last_status"] = r["status"]

                # Check completion based on pipeline type
                # A resume-skip of the final step is as healthy as a fresh success:
                # its outputs exist from a prior completed run.
                if (
                    r["step"] == final_step
                    and r["status"].lower() in ("success", "skipped_complete")
                ):
                    current["complete"] = True

                summary[key] = current

    return list(summary.values())


def _sort_key(row: dict) -> tuple:
    return (row["target"], row["band"], row["night"], row["pipeline"])


def print_table(rows: list[dict]):
    print(
        format_table(
            sorted(rows, key=_sort_key),
            columns=[
                ("TARGET", lambda r: r["target"]),
                ("INSTR", lambda r: r["instrument"]),
                ("BAND", lambda r: r["band"]),
                ("NIGHT", lambda r: r["night"]),
                ("PIPELINE", lambda r: r["pipeline"]),
                ("COMPLETE", lambda r: str(r["complete"])),
                ("LAST_STEP", lambda r: r["last_step"]),
                ("STATUS", lambda r: r["last_status"]),
            ],
        )
    )


def write_csv(rows: list[dict], path: Path):
    fieldnames = [
        "target",
        "instrument",
        "band",
        "night",
        "pipeline",
        "complete",
        "last_step",
        "last_status",
        "log_path",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=_sort_key))
    print(f"\nCSV summary written to {path}", file=sys.stderr)


def main():
    args = parse_args()
    rows = aggregate(args.root_dir)

    if args.pipeline != "all":
        rows = [r for r in rows if r["pipeline"] == args.pipeline]
    if args.instrument != "all":
        rows = [r for r in rows if r["instrument"].lower() == args.instrument]

    if not rows:
        print("⚠️  No structured log entries found.", file=sys.stderr)
        sys.exit(1)

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()

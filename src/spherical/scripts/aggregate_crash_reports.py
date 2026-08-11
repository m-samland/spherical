#!/usr/bin/env python
"""
aggregate_crash_reports.py

Summarise crash reports produced by the SPHERE/IFS pipeline.

Usage
-----
    python aggregate_crash_reports.py /path/to/reductions [--csv crashes.csv] [--top N]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from spherical.scripts._monitoring import format_table, instrument_from_band

# The reduction and TRAP crash reports open with different sentences, so both
# headers must be recognised or TRAP datasets come out as "unknown".
DATASET_PATTERNS = (
    re.compile(r"reduction of (.+?)\.\s*$"),          # IFS / IRDIS reduction
    re.compile(r"TRAP processing error for (.+?)\s*$"),  # TRAP post-processing
)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise pipeline crash reports.")
    p.add_argument("root_dir", type=Path,
                   help="Directory tree containing crash_report.txt files")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional path to write summary table")
    p.add_argument("--top", type=int, default=10,
                   help="Show N most frequent exception types (default 10)")
    p.add_argument("--instrument", choices=["ifs", "irdis", "all"], default="all",
                   help="Filter by instrument (default: all). Reports whose dataset "
                        "identifier could not be parsed are dropped by any filter.")
    return p.parse_args()


def extract_dataset(header: str) -> str:
    """Return the ``target/band/night`` identifier from a crash-report header."""
    for pattern in DATASET_PATTERNS:
        m = pattern.search(header)
        if m:
            return m.group(1)
    return "unknown"


def extract_instrument(dataset: str) -> str:
    """Return the instrument for a ``target/band/night`` dataset identifier."""
    parts = dataset.split("/")
    return instrument_from_band(parts[1]) if len(parts) == 3 else "unknown"


def extract_info(report: Path) -> dict:
    """
    Return a dict with:
        dataset   – dataset identifier parsed from first line
        exc_type  – Exception class name
        exc_msg   – Short exception message
        first_file – file:line string of the deepest stack frame
        report_path – path to crash_report.txt
    """
    with report.open() as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    dataset = extract_dataset(lines[0]) if lines else "unknown"

    # Trim empty lines at end and walk back to find last non-empty traceback line
    tlines = [ln for ln in lines if ln.strip()]
    exc_type = exc_msg = first_file = "unknown"
    if len(tlines) >= 2:
        # Last line: "ValueError: invalid shape ..."
        exc_line = tlines[-1]
        if ":" in exc_line:
            exc_type, exc_msg = exc_line.split(":", 1)
            exc_type, exc_msg = exc_type.strip(), exc_msg.strip()
        else:
            exc_type = exc_line.strip()

        # Last "File ..." line just before the exception line
        for ln in reversed(tlines[:-1]):
            if ln.strip().startswith("File"):
                first_file = ln.strip()
                break

    return {
        "dataset": dataset,
        "instrument": extract_instrument(dataset),
        "exc_type": exc_type,
        "exc_msg": exc_msg,
        "first_file": first_file,
        "report_path": str(report),
        "pipeline": "trap" if "trap" in report.name.lower() else "reduction",
    }


def print_table(rows: list[dict]):
    print(
        format_table(
            sorted(rows, key=lambda x: x["dataset"]),
            columns=[
                ("DATASET", lambda r: r["dataset"]),
                ("INSTR", lambda r: r["instrument"]),
                ("PIPELINE", lambda r: r["pipeline"]),
                ("EXCEPTION", lambda r: r["exc_type"]),
            ],
            trailing=("MESSAGE", lambda r: r["exc_msg"]),
        )
    )


def write_csv(rows: list[dict], path: Path):
    fieldnames = ["dataset", "instrument", "pipeline", "exc_type", "exc_msg", "first_file", "report_path"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV summary written to {path}", file=sys.stderr)


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    args = parse_args()
    rows: list[dict] = []

    # Search for both IFS and TRAP crash reports
    for pattern in ["crash_report.txt", "trap_crash_report.txt"]:
        for rpt in args.root_dir.rglob(pattern):
            rows.append(extract_info(rpt))

    if not rows:
        print("✅ No crash reports found.")
        return

    if args.instrument != "all":
        rows = [r for r in rows if r["instrument"].lower() == args.instrument]
        if not rows:
            print(f"✅ No crash reports found for instrument '{args.instrument}'.")
            return

    # Print table
    print_table(rows)

    # Optional CSV
    if args.csv:
        write_csv(rows, args.csv)

    # Frequency of exceptions
    counts = Counter(r["exc_type"] for r in rows)
    print("\n🔢 Most frequent exceptions:")
    for exc, n in counts.most_common(args.top):
        print(f"{exc}: {n}")


if __name__ == "__main__":
    main()

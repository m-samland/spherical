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
    return p.parse_args()


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

    # Dataset name (first line e.g. "An error occurred during the reduction of bet_Pic/OBS_H/2014-12-07.")
    dataset = "unknown"
    if lines and "reduction of" in lines[0]:
        m = re.search(r"reduction of (.+?)\.", lines[0])
        if m:
            dataset = m.group(1)

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
        "exc_type": exc_type,
        "exc_msg": exc_msg,
        "first_file": first_file,
        "report_path": str(report),
    }


def print_table(rows: list[dict]):
    pad = 28
    hdr = (f'{"DATASET":{pad}} {"EXCEPTION":22} {"MESSAGE"}')
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["dataset"]):
        print(f'{r["dataset"]:{pad}} {r["exc_type"][:21]:22} {r["exc_msg"]}')


def write_csv(rows: list[dict], path: Path):
    fieldnames = ["dataset", "exc_type", "exc_msg", "first_file", "report_path"]
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

    for rpt in args.root_dir.rglob("crash_report.txt"):
        rows.append(extract_info(rpt))

    if not rows:
        print("✅ No crash reports found.")
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

#!/usr/bin/env python
"""
aggregate_reduction_status.py

Summarise SPHERE/IFS reduction status across many JSON logs.

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

# Final steps for different pipeline types
FINAL_STEPS = {
    "ifs": "spot_to_flux_normalization",
    "trap": "trap_session"  # TRAP session completion
}

def detect_pipeline_type(jsonlog_path: Path) -> str:
    """Detect pipeline type from log file name or path."""
    if "trap" in jsonlog_path.name.lower():
        return "trap"
    return "ifs"


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
        "--pipeline-type", 
        choices=["ifs", "trap", "all"], 
        default="all",
        help="Filter by pipeline type (default: all)"
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
        'band': str,
        'night': str,
        'pipeline_type': str,
        'complete': bool,
        'last_step': str,
        'last_status': str,
        'log_path': str
    }
    """
    summary: dict[tuple, dict] = {}

    # Search for both IFS and TRAP JSON logs
    for pattern in ["reduction.jsonlog", "trap_reduction.jsonlog"]:
        for jsonlog in root.rglob(pattern):
            pipeline_type = detect_pipeline_type(jsonlog)
            final_step = FINAL_STEPS[pipeline_type]
            
            rows = extract_structured_rows(jsonlog)
            for r in rows:
                key = (r["target"], r["band"], r["night"], pipeline_type)  # Include pipeline type in key
                current = summary.get(
                    key,
                    {
                        "target": r["target"],
                        "band": r["band"],
                        "night": r["night"],
                        "pipeline_type": pipeline_type,  # New field
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
                if (
                    r["step"] == final_step
                    and r["status"].lower() == "success"
                ):
                    current["complete"] = True

                summary[key] = current

    return list(summary.values())


def print_table(rows: list[dict]):
    pad = 14
    header = (
        f'{"TARGET":{pad}} {"BAND":4} {"NIGHT":12} {"TYPE":4} '
        f'{"COMPLETE":9} {"LAST_STEP":26} {"STATUS":8}'
    )
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (x["target"], x["band"], x["night"], x["pipeline_type"])):
        print(
            f'{r["target"]:{pad}} {r["band"]:4} {r["night"]:12} {r["pipeline_type"]:4} '
            f'{str(r["complete"]):9} {r["last_step"][:24]:26} {r["last_status"]}'
        )


def write_csv(rows: list[dict], path: Path):
    fieldnames = [
        "target",
        "band",
        "night",
        "pipeline_type",  # New field
        "complete",
        "last_step",
        "last_status",
        "log_path",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV summary written to {path}", file=sys.stderr)


def main():
    args = parse_args()
    rows = aggregate(args.root_dir)
    
    # Filter by pipeline type if specified
    if args.pipeline_type != "all":
        rows = [r for r in rows if r["pipeline_type"] == args.pipeline_type]
    
    if not rows:
        print("⚠️  No structured log entries found.", file=sys.stderr)
        sys.exit(1)

    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()

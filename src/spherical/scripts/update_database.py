"""CLI: bring the SPHERE database tables up to date (or re-enrich them).

Extends the file table from the last covered night to today, then rebuilds the
target and observation tables for every mode (including SAM). With
``--enrich-only`` it refreshes Gaia/MOCA enrichment on existing tables without
querying ESO.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spherical.database import build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update or re-enrich the spherical database tables.")
    parser.add_argument("--dest", type=Path, required=True, help="Directory holding the database tables.")
    parser.add_argument("--instrument", choices=("ifs", "irdis", "all"), default="all")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD; default: derived from provenance minus overlap.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD; default: today.")
    parser.add_argument("--overlap-days", type=int, default=7, help="Days to back off from the last covered night.")
    parser.add_argument("--suffix", default="", help="Filename suffix for experimental/scratch builds.")
    parser.add_argument("--skip-sam", action="store_true", help="Do not build SAM tables.")
    parser.add_argument("--no-enrich", dest="enrich", action="store_false", help="Skip Gaia/MOCA enrichment.")
    parser.add_argument("--enrich-only", action="store_true", help="Only re-run Gaia/MOCA enrichment; do not query ESO.")
    return parser


def _instruments(choice):
    return ["ifs", "irdis"] if choice == "all" else [choice]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        for instrument in _instruments(args.instrument):
            if args.enrich_only:
                for mode_kw in build.modes_for_instrument(instrument, skip_sam=args.skip_sam):
                    build.enrich_tables(
                        args.dest,
                        instrument,
                        polarimetry=mode_kw["polarimetry"],
                        sparse_aperture_masking=mode_kw["sparse_aperture_masking"],
                        suffix=args.suffix,
                    )
            else:
                build.update_database(
                    args.dest,
                    instrument,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    overlap_days=args.overlap_days,
                    suffix=args.suffix,
                    skip_sam=args.skip_sam,
                    enrich=args.enrich,
                )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI: bring the SPHERE database tables up to date (or re-enrich them).

Extends the file table from the last covered night to today, then rebuilds the
target and observation tables for every mode (including SAM). With
``--enrich-only`` it refreshes Gaia/MOCA enrichment on existing tables without
querying ESO. ``--mode`` narrows an ``--enrich-only`` run to a single mode
(e.g. ``irdis``), which is handy for retrying one mode after a transient
Gaia/MOCA outage without redoing the others.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spherical.database import build
from spherical.database.database_utils import resolve_mode_name

MODE_CHOICES = ("ifs", "ifs_sam", "irdis", "irdis_polarimetry", "irdis_sam")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update or re-enrich the spherical database tables.")
    parser.add_argument("--dest", type=Path, required=True, help="Directory holding the database tables.")
    parser.add_argument("--instrument", choices=("ifs", "irdis", "all"), default="all")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=None,
        help="Re-enrich only this single mode (requires --enrich-only). Useful to retry one mode without redoing the others.",
    )
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


def _resolve_single_mode(mode_name):
    """Map a canonical mode name back to (instrument, mode_kwargs)."""
    instrument = "ifs" if mode_name.startswith("ifs") else "irdis"
    for mode_kw in build.modes_for_instrument(instrument, skip_sam=False):
        name = resolve_mode_name(instrument, mode_kw["polarimetry"], mode_kw["sparse_aperture_masking"])
        if name == mode_name:
            return instrument, mode_kw
    raise ValueError(f"Unknown mode: {mode_name}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.mode is not None:
            if not args.enrich_only:
                print("Error: --mode is only supported together with --enrich-only.", file=sys.stderr)
                return 1
            instrument, mode_kw = _resolve_single_mode(args.mode)
            build.enrich_tables(
                args.dest,
                instrument,
                polarimetry=mode_kw["polarimetry"],
                sparse_aperture_masking=mode_kw["sparse_aperture_masking"],
                suffix=args.suffix,
            )
            return 0

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

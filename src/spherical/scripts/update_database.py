"""CLI: bring the SPHERE database tables up to date (or re-enrich them).

Extends the file table from the last covered night to today, then rebuilds the
target and observation tables for every mode (including SAM). With
``--enrich-only`` it refreshes Gaia/MOCA enrichment on existing tables without
querying ESO. ``--mode`` narrows an ``--enrich-only`` run to a single mode
(e.g. ``irdis``), which is handy for retrying one mode after a transient
Gaia/MOCA outage without redoing the others.

After the tables are written, a per-mode Gaia/MOCA enrichment health summary is
printed. Each source is compared against absolute match-fraction floors and the
previous run's fractions (see ``spherical.database.enrichment_health``); the CLI
exits non-zero if any mode's enrichment failed or degraded, so unattended runs
surface a likely-failed enrichment rather than silently succeeding.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spherical.database import build
from spherical.database import enrichment_health as eh
from spherical.database import provenance as prov
from spherical.database.database_utils import resolve_mode_name

logger = logging.getLogger("spherical.update")

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


def _modes_processed(args):
    """Canonical mode names the run (re)built, used to scope the health summary."""
    if args.mode is not None:
        return [args.mode]
    modes = []
    for instrument in _instruments(args.instrument):
        for mode_kw in build.modes_for_instrument(instrument, skip_sam=args.skip_sam):
            modes.append(resolve_mode_name(instrument, mode_kw["polarimetry"], mode_kw["sparse_aperture_masking"]))
    return modes


def _summarize_enrichment(dest, prev, modes) -> int:
    """Print a per-mode Gaia/MOCA health summary; return the number of breaches.

    Each (re)built mode's stored metrics are compared against the pre-run
    provenance (``prev``) using the floor + regression policy in
    ``enrichment_health``. A breach is a failed/degraded source.
    """
    new = prov.read_provenance(dest)
    rows = []
    breaches = 0
    for mode in modes:
        record = new.get(mode)
        if record is None or not isinstance(record.enrichment, dict):
            continue
        prev_enrich = prev[mode].enrichment if mode in prev else {}
        for source in ("gaia", "moca"):
            metrics = record.enrichment.get(source)
            if not isinstance(metrics, dict):
                continue
            prev_metrics = prev_enrich.get(source) if isinstance(prev_enrich, dict) else None
            verdict = eh.evaluate(source, metrics, prev_metrics)
            status = metrics.get("status", "?")
            frac = metrics.get("frac")
            frac_s = f"{frac * 100:4.0f}%" if isinstance(frac, (int, float)) else "  --"
            if status == "skipped":
                tag = "SKIP"
            elif verdict.passed:
                tag = "OK"
            else:
                tag = "WARN"
                breaches += 1
                for reason in verdict.reasons:
                    logger.warning("enrichment health: %s/%s: %s", mode, source, reason)
            rows.append((mode, source, frac_s, tag, verdict.reasons))

    if rows:
        print("\n=== enrichment health summary ===", file=sys.stderr)
        for mode, source, frac_s, tag, reasons in rows:
            detail = f"  ({'; '.join(reasons)})" if reasons else ""
            print(f"  {mode:<20} {source:<5} {frac_s}  {tag}{detail}", file=sys.stderr)
    return breaches


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    prev = prov.read_provenance(args.dest)

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
        else:
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

    breaches = _summarize_enrichment(args.dest, prev, _modes_processed(args))
    return 1 if breaches else 0


if __name__ == "__main__":
    raise SystemExit(main())

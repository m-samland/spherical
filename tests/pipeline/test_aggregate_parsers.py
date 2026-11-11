"""The aggregate scripts expose build_parser() for the CLI reference page."""

from pathlib import Path

from spherical.scripts import aggregate_crash_reports, aggregate_reduction_status


def test_crash_reports_parser():
    args = aggregate_crash_reports.build_parser().parse_args(["/runs", "--top", "3", "--instrument", "irdis"])
    assert (args.root_dir, args.top, args.instrument, args.csv) == (Path("/runs"), 3, "irdis", None)


def test_reduction_status_parser():
    args = aggregate_reduction_status.build_parser().parse_args(["/runs", "--pipeline", "trap"])
    assert (args.root_dir, args.pipeline, args.instrument) == (Path("/runs"), "trap", "all")

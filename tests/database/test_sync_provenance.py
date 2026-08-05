import csv

from spherical.scripts import sync_zenodo_tables as sz


def _write_file_table(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["DP.ID", "NIGHT_START"])
        w.writerow(["a", "2016-09-15"])
        w.writerow(["b", "2026-06-30"])
        w.writerow(["c", "2014-05-01"])


def test_compute_coverage(tmp_path):
    csv_path = tmp_path / "table_of_files_ifs.csv"
    _write_file_table(csv_path)
    start, end = sz.compute_coverage_from_file_table(csv_path)
    assert start == "2014-05-01"
    assert end == "2026-06-30"


def test_compute_coverage_missing_file(tmp_path):
    start, end = sz.compute_coverage_from_file_table(tmp_path / "missing.csv")
    assert start is None and end is None

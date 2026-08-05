"""The monitoring scripts must report the right dataset and instrument for both
instruments and both pipelines."""
import json

from spherical.scripts._monitoring import format_table, instrument_from_band
from spherical.scripts.aggregate_crash_reports import extract_info
from spherical.scripts.aggregate_reduction_status import aggregate, detect_pipeline

TRAP_REPORT = """TRAP processing error for *_bet_Pic/OBS_H/2019-03-10

Error: SVD did not converge

Traceback (most recent call last):
  File "/pkg/run_trap.py", line 1, in run
    raise numpy.linalg.LinAlgError("SVD did not converge")
numpy.linalg.LinAlgError: SVD did not converge
"""

REDUCTION_REPORT = """An error occurred during the reduction of *_51_Eri/DB_K12/2017-09-27.

Traceback (most recent call last):
  File "/pkg/irdis_reduction.py", line 1, in run
    raise ValueError("Empty filename: ''")
ValueError: Empty filename: ''
"""


def _write_log(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _rec(step, status, band="OBS_H"):
    return {
        "target": "HD1",
        "band": band,
        "night": "2020-01-01",
        "step": step,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# instrument attribution
# --------------------------------------------------------------------------- #
def test_ifs_bands_map_to_ifs():
    assert instrument_from_band("OBS_YJ") == "IFS"
    assert instrument_from_band("OBS_H") == "IFS"


def test_irdis_bands_map_to_irdis():
    for band in ("DB_K12", "BB_H", "NB_CntH", "DP_0_BB_J"):
        assert instrument_from_band(band) == "IRDIS"


# --------------------------------------------------------------------------- #
# reduction_status
# --------------------------------------------------------------------------- #
def test_detect_pipeline_from_log_name(tmp_path):
    assert detect_pipeline(tmp_path / "reduction.jsonlog") == "reduction"
    assert detect_pipeline(tmp_path / "trap_reduction.jsonlog") == "trap"


def test_irdis_reduction_reports_irdis_and_completes(tmp_path):
    """IRDIS shares the IFS final-step log name, so completion must be detected
    the same way, but the row must not be labelled IFS."""
    _write_log(
        tmp_path / "a" / "reduction.jsonlog",
        [_rec("spot_to_flux_normalization", "success", band="DB_K12")],
    )
    (row,) = aggregate(tmp_path)
    assert row["instrument"] == "IRDIS"
    assert row["pipeline"] == "reduction"
    assert row["complete"] is True


def test_trap_row_carries_the_observation(tmp_path):
    _write_log(
        tmp_path / "b" / "trap_reduction.jsonlog",
        [_rec("trap_session", "success", band="DB_K12")],
    )
    (row,) = aggregate(tmp_path)
    assert (row["target"], row["band"], row["night"]) == ("HD1", "DB_K12", "2020-01-01")
    assert row["pipeline"] == "trap"
    assert row["complete"] is True


def test_reduction_and_trap_logs_are_separate_rows(tmp_path):
    _write_log(tmp_path / "c" / "reduction.jsonlog", [_rec("spot_to_flux_normalization", "success")])
    _write_log(tmp_path / "c" / "trap_reduction.jsonlog", [_rec("trap_session", "success")])
    rows = aggregate(tmp_path)
    assert sorted(r["pipeline"] for r in rows) == ["reduction", "trap"]


# --------------------------------------------------------------------------- #
# crash_reports
# --------------------------------------------------------------------------- #
def test_trap_crash_report_dataset_is_parsed(tmp_path):
    report = tmp_path / "trap_crash_report.txt"
    report.write_text(TRAP_REPORT)
    info = extract_info(report)
    assert info["dataset"] == "*_bet_Pic/OBS_H/2019-03-10"
    assert info["instrument"] == "IFS"
    assert info["pipeline"] == "trap"
    assert info["exc_type"] == "numpy.linalg.LinAlgError"


def test_reduction_crash_report_dataset_is_parsed(tmp_path):
    report = tmp_path / "crash_report.txt"
    report.write_text(REDUCTION_REPORT)
    info = extract_info(report)
    assert info["dataset"] == "*_51_Eri/DB_K12/2017-09-27"
    assert info["instrument"] == "IRDIS"
    assert info["pipeline"] == "reduction"


def test_unrecognised_header_stays_unknown(tmp_path):
    report = tmp_path / "crash_report.txt"
    report.write_text("Something went wrong\n\nValueError: nope\n")
    info = extract_info(report)
    assert info["dataset"] == "unknown"
    assert info["instrument"] == "unknown"


# --------------------------------------------------------------------------- #
# table formatting
# --------------------------------------------------------------------------- #
def test_columns_stay_aligned_when_values_exceed_the_header():
    rows = [
        {"target": "2MASS_J19232412-0740113", "band": "DB_K12"},
        {"target": "HD1", "band": "OBS_H"},
    ]
    table = format_table(
        rows,
        columns=[("TARGET", lambda r: r["target"]), ("BAND", lambda r: r["band"])],
    )
    header, rule, *body = table.splitlines()
    assert len(rule) == len(header)
    assert header.index("BAND") == body[0].index("DB_K12")
    assert header.index("BAND") == body[1].index("OBS_H")

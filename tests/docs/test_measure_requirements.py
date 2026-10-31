"""Step times are read correctly from real log patterns; disk use is counted per stage."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "tools"))
import measure_requirements as mr  # noqa: E402


def rec(t, step, status, msg="m"):
    return json.dumps({"asctime": f"2026-10-20 10:{t},000", "levelname": "INFO",
                       "message": msg, "step": step, "status": status})


def write_log(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def test_pairs_started_with_success(tmp_path):
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "session_start", "started"),
        rec("00:05", "download_data", "started"), rec("02:05", "download_data", "success")])
    steps, sessions = mr.read_step_times([log])
    assert sessions == 1 and [(s.step, s.seconds) for s in steps] == [("download_data", 120.0)]


def test_failed_warning_does_not_close_step(tmp_path):
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "frame_info_computation", "started"),
        rec("00:01", "frame_info_computation", "failed", "No data for frame type: CORO"),
        rec("00:10", "frame_info_computation", "success")])
    (step,), _ = mr.read_step_times([log])
    assert step.seconds == 10.0


def test_repeated_started_is_one_interval(tmp_path):
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "extract_cubes", "started"), rec("00:20", "extract_cubes", "started"),
        rec("09:00", "extract_cubes", "success")])
    (step,), _ = mr.read_step_times([log])
    assert step.seconds == 540.0


@pytest.mark.parametrize("status", ["skipped", "skipped_complete"])
def test_skipped_closes_step(tmp_path, status):
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "extract_cubes", "started"), rec("00:02", "extract_cubes", status)])
    (step,), _ = mr.read_step_times([log])
    assert step.seconds == 2.0


def test_trap_session_is_a_marker(tmp_path):
    log = write_log(tmp_path / "trap_reduction.jsonlog", [
        rec("00:00", "trap_session_start", "started"),
        rec("00:01", "trap_reduction", "started"), rec("30:01", "trap_reduction", "success"),
        rec("30:02", "trap_session", "success")])
    steps, sessions = mr.read_step_times([log])
    assert [s.step for s in steps] == ["trap_reduction"] and sessions == 1


def test_unfinished_step_is_reported(tmp_path):
    log = write_log(tmp_path / "reduction.jsonlog", [rec("00:00", "extract_cubes", "started")])
    (step,), _ = mr.read_step_times([log])
    assert step.end is None and step.seconds is None


def test_unclosed_step_ends_when_the_next_step_starts(tmp_path):
    # IRDIS polynomial_center_fit logs `started` but never `success` (process_centers.py).
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "polynomial_center_fit", "started"),
        rec("00:07", "plot_center_evolution", "started"), rec("00:08", "plot_center_evolution", "success")])
    fit, plot = mr.read_step_times([log])[0]
    assert (fit.seconds, fit.approximate) == (7.0, True)
    assert (plot.seconds, plot.approximate) == (1.0, False)


def test_nested_step_gets_depth_one(tmp_path):
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "find_centers", "started"), rec("00:01", "fit_centers", "started"),
        rec("00:09", "fit_centers", "success"), rec("00:10", "find_centers", "success")])
    steps, _ = mr.read_step_times([log])
    assert {s.step: s.depth for s in steps} == {"find_centers": 0, "fit_centers": 1}


def test_rotated_files_are_merged_in_time_order(tmp_path):
    old = write_log(tmp_path / "reduction.jsonlog.1", [rec("00:00", "a", "started")])
    new = write_log(tmp_path / "reduction.jsonlog", [rec("00:30", "a", "success")])
    (step,), _ = mr.read_step_times([new, old])
    assert step.seconds == 30.0


def test_multiple_sessions_are_counted(tmp_path):
    log = write_log(tmp_path / "reduction.jsonlog", [
        rec("00:00", "session_start", "started"), rec("05:00", "session_start", "started")])
    assert mr.read_step_times([log])[1] == 2


def test_null_step_and_bad_lines_are_ignored(tmp_path):
    log = tmp_path / "reduction.jsonlog"
    log.write_text(rec("00:00", None, None) + "\nnot json\n")
    assert mr.read_step_times([log]) == ([], 0)


def test_directory_bytes_skips_symlinks_and_missing(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_bytes(b"x" * 10)
    (tmp_path / "d" / "link").symlink_to(tmp_path / "d" / "f")
    assert mr.directory_bytes(tmp_path / "d") == 10
    assert mr.directory_bytes(tmp_path / "missing") == 0


def test_stage_paths_ifs(tmp_path):
    stages = mr.stage_paths("ifs", tmp_path / "raw", tmp_path / "red", "*_51_Eri", "OBS_H", "2015-09-24")
    assert set(stages) == {"raw_science", "raw_calibration", "calibration", "extracted", "products", "trap"}
    assert stages["products"] == [tmp_path / "red/IFS/observation/*_51_Eri/OBS_H/2015-09-24/optext/converted"]


def test_stage_paths_irdis_has_no_extracted(tmp_path):
    stages = mr.stage_paths("irdis", tmp_path / "raw", tmp_path / "red", "*_51_Eri", "DB_K12", "2015-09-24")
    assert "extracted" not in stages


def test_write_rows_replaces_only_same_label(tmp_path):
    csv_path = tmp_path / "t.csv"
    mr.write_rows(csv_path, ("irdis", "51eri_irdis"), [{"step": "a", "seconds": 1}])
    mr.write_rows(csv_path, ("irdis", "51eri_irdis_ncpu2"), [{"step": "a", "seconds": 2}])
    mr.write_rows(csv_path, ("irdis", "51eri_irdis"), [{"step": "b", "seconds": 3}])
    text = csv_path.read_text()
    assert "51eri_irdis,b" in text and "51eri_irdis_ncpu2,a" in text and "51eri_irdis,a," not in text


def test_frame_counts_exclude_header(tmp_path):
    for kind, n in (("coro", 3), ("center", 1), ("flux", 2)):
        (tmp_path / f"frames_info_{kind}.csv").write_text("a,b\n" + "1,2\n" * n)
    assert mr.frame_counts(tmp_path) == {"frames_coro": 3, "frames_center": 1, "frames_flux": 2}


def _tree(tmp_path, monkeypatch, with_products=True):
    monkeypatch.setenv("SPHERICAL_TUTORIAL_DIR", str(tmp_path))
    obs = tmp_path / "lab" / "reduction/IRDIS/observation/*_51_Eri/DB_K12/2015-09-24"
    obs.mkdir(parents=True)
    write_log(obs / "reduction.jsonlog", [rec("00:00", "session_start", "started")])
    if with_products:
        (obs / "converted").mkdir()
    return ["--instrument", "irdis", "--label", "lab", "--target", "*_51_Eri", "--filter", "DB_K12",
            "--date", "2015-09-24", "--out-dir", str(tmp_path / "csv"), "--allow-partial"]


def test_missing_run_is_refused_even_when_partial_is_allowed(tmp_path, monkeypatch):
    args = _tree(tmp_path, monkeypatch)
    args[args.index("lab")] = "typo"
    assert mr.main(args) == 1
    assert not (tmp_path / "csv").exists()


def test_missing_products_are_refused(tmp_path, monkeypatch):
    assert mr.main(_tree(tmp_path, monkeypatch, with_products=False)) == 1


def test_complete_tree_is_measured(tmp_path, monkeypatch):
    assert mr.main(_tree(tmp_path, monkeypatch)) == 0
    assert (tmp_path / "csv" / "requirements_51eri_disk.csv").exists()

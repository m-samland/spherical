"""reduction_status must treat a resume-skip as healthy/complete."""
import json

from spherical.scripts.aggregate_reduction_status import aggregate


def _write_log(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _rec(step, status):
    return {"target": "HD1", "band": "OBS_H", "night": "2020-01-01", "step": step, "status": status}


def test_skipped_complete_final_step_is_complete(tmp_path):
    _write_log(
        tmp_path / "a" / "reduction.jsonlog",
        [_rec("extract_cubes", "skipped_complete"), _rec("spot_to_flux_normalization", "skipped_complete")],
    )
    rows = aggregate(tmp_path)
    assert len(rows) == 1
    assert rows[0]["complete"] is True


def test_failed_final_step_is_incomplete(tmp_path):
    _write_log(
        tmp_path / "b" / "reduction.jsonlog",
        [_rec("spot_to_flux_normalization", "failed")],
    )
    rows = aggregate(tmp_path)
    assert rows[0]["complete"] is False


def test_plain_skipped_final_step_is_incomplete(tmp_path):
    _write_log(
        tmp_path / "c" / "reduction.jsonlog",
        [_rec("spot_to_flux_normalization", "skipped")],
    )
    rows = aggregate(tmp_path)
    assert rows[0]["complete"] is False

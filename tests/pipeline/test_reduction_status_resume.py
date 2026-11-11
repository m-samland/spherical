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


def test_leaf_step_after_the_final_step_does_not_reopen_completion(tmp_path):
    """align_frames runs after spot_to_flux, so it is the last step in the log
    of a run that enables it. Completion is owned by the final step; a leaf that
    follows it must not flip a finished reduction back to incomplete."""
    _write_log(
        tmp_path / "d" / "reduction.jsonlog",
        [_rec("spot_to_flux_normalization", "success"), _rec("frame_alignment", "success")],
    )
    rows = aggregate(tmp_path)
    assert rows[0]["complete"] is True
    assert rows[0]["last_step"] == "frame_alignment"


def test_failed_leaf_step_leaves_the_reduction_complete(tmp_path):
    """A leaf feeds nothing downstream, so its failure does not invalidate the
    reduction. The row stays COMPLETE=True and the failure is visible in STATUS."""
    _write_log(
        tmp_path / "e" / "reduction.jsonlog",
        [_rec("spot_to_flux_normalization", "success"), _rec("frame_alignment", "failed")],
    )
    rows = aggregate(tmp_path)
    assert rows[0]["complete"] is True
    assert rows[0]["last_status"] == "failed"

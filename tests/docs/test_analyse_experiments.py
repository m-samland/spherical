"""The experiment analysis reproduces TRAP's shape check and labels candidates correctly."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "tutorials" / "runs" / "investigation"))
import analyse_experiments as ae  # noqa: E402


@pytest.mark.parametrize("theta, x, y", [(166.0, 3.0, 4.5), (256.0, 4.5, 3.0), (-14.0, 3.0, 4.5)])
def test_canonical_shape_is_the_same_for_every_parametrisation(theta, x, y):
    # One radially elongated blob at PA 166, written three ways by a free Gaussian fit.
    m = ae.shape_metrics(166.0, theta, x, y)
    assert m["canonical_dev"] == pytest.approx(0.0, abs=1e-9)
    assert m["elongation"] == pytest.approx(1.5)


def test_trap_deviation_depends_on_the_parametrisation():
    assert ae.shape_metrics(166.0, 166.0, 3.0, 4.5)["trap_dev"] == pytest.approx(0.0)
    assert abs(ae.shape_metrics(166.0, -14.0, 3.0, 4.5)["trap_dev"]) == pytest.approx(180.0)
    assert ae.shape_metrics(166.0, 256.0, 4.5, 3.0)["ratio"] < 1.0


def test_trap_deviation_matches_the_tutorial_candidate():
    # 2015 tutorial run, T-type: TRAP wrote theta_deviation -33.03 and ratio 0.617.
    m = ae.shape_metrics(165.59774870652973, -161.36762837980484, 5.106170852888713, 3.151166654)
    assert m["trap_dev"] == pytest.approx(-33.0346, abs=1e-3)
    assert m["ratio"] == pytest.approx(0.6171, abs=1e-3)


def _candidate(**changes):
    c = {"snr": 6.0, "good_fraction": 1.0, "trap_dev": 10.0, "ratio": 1.3, "canonical_dev": 10.0, "elongation": 1.3}
    return c | changes


def test_rule_sets():
    rules = ae.rule_sets()
    assert all(rule(_candidate()) for rule in rules.values())
    swapped = _candidate(trap_dev=-80.0, ratio=1 / 1.3)
    assert not rules["trap"](swapped) and rules["canonical"](swapped)
    assert rules["canonical_35"](_candidate(canonical_dev=30.0)) and not rules["canonical"](_candidate(canonical_dev=30.0))
    assert rules["elong_1.0"](_candidate(elongation=1.05)) and not rules["canonical"](_candidate(elongation=1.05))
    assert not rules["no_shape"](_candidate(snr=4.9))


def test_targets_and_labels():
    known = ae.targets("IRDIS", "2015-09-24", {"fakes": [{"id": 0, "yx": [10.0, -20.0]}]})
    assert [label for label, _ in known] == ["b", "fake0"]
    b_y, b_x = known[0][1]
    assert b_y == pytest.approx(-36.20, abs=0.01) and b_x == pytest.approx(-8.46, abs=0.01)
    assert ae.label_candidate({"y_relative": -36.0, "x_relative": -9.0}, known) == "b"
    assert ae.label_candidate({"y_relative": 11.0, "x_relative": -21.0}, known) == "fake0"
    assert ae.label_candidate({"y_relative": 0.0, "x_relative": 30.0}, known) == "other"


COLUMNS = ["candidate_id", "x_relative", "y_relative", "separation", "position_angle", "theta_free",
           "x_fwhm_free", "y_fwhm_free", "good_fraction_free", "norm_snr_fit_free", "peak_pixel_snr",
           "theta_deviation", "wavelength_index"]


def _write_result(root, rows, validated):
    tm = root / "reduction" / "IRDIS" / "trap" / "*_51_Eri" / "DB_K12" / "2015-09-24" / "template_matching"
    tm.mkdir(parents=True)
    with (tm / "companion_table_T-type.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            for channel in (0, 1):  # TRAP repeats each candidate once per channel
                writer.writerow(row | {"wavelength_index": channel})
    with (tm / "validated_companion_table_T-type.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id"])
        writer.writeheader()
        writer.writerows({"candidate_id": cid} for cid in validated)
    return tm.parent


def test_analyse_counts_and_checks_against_trap(tmp_path):
    exp = tmp_path / "exp"
    b = {"candidate_id": "0", "x_relative": -9.28, "y_relative": -36.15, "separation": 37.3, "position_angle": 165.6,
         "theta_free": -161.4, "x_fwhm_free": 5.1, "y_fwhm_free": 3.15, "good_fraction_free": 1.29,
         "norm_snr_fit_free": 5.97, "peak_pixel_snr": 5.8, "theta_deviation": -33.0}
    fake = b | {"candidate_id": "1", "x_relative": 30.0, "y_relative": 0.0, "position_angle": 270.0,
                "theta_free": 270.0, "x_fwhm_free": 3.0, "y_fwhm_free": 4.5, "theta_deviation": 0.0}
    _write_result(exp, [b, fake], validated=["1"])
    (exp / "experiment.json").write_text(json.dumps({"name": "e", "fakes": [{"id": 0, "yx": [0.0, 30.0]}]}))
    rows, summary = ae.analyse([exp], ae.rule_sets())
    assert len(rows) == 2 and {r["label"] for r in rows} == {"b", "fake0"}
    trap = next(s for s in summary if s["rule"] == "trap")
    assert (trap["b_passes"], trap["fakes_passing"], trap["fakes_injected"], trap["trap_rule_mismatch"]) == (False, 1, 1, "")


def test_analyse_flags_a_disagreement_with_trap(tmp_path):
    row = {"candidate_id": "0", "x_relative": 0.0, "y_relative": 30.0, "separation": 30.0, "position_angle": 0.0,
           "theta_free": 0.0, "x_fwhm_free": 3.0, "y_fwhm_free": 4.5, "good_fraction_free": 1.0,
           "norm_snr_fit_free": 8.0, "peak_pixel_snr": 8.0, "theta_deviation": 0.0}
    folder = _write_result(tmp_path / "exp", [row], validated=[])
    _rows, summary = ae.analyse([folder], ae.rule_sets())
    assert next(s for s in summary if s["rule"] == "trap")["trap_rule_mismatch"] == "0"

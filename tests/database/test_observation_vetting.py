"""create_observation_table with file ownership and vetting (#224), on synthetic tables."""

import numpy as np
import pytest
from astropy.table import Table

from spherical.database import match_vetting as mv
from spherical.database.observation_table import OTHER_LEADING_COLUMNS, create_observation_table

NIGHT = "2016-04-16"
ARCSEC = 1 / 3600.0


def _file_rows(obj, ra, dec, n=6, start=57495.2, drift_arcsec_per_frame=0.0, targ=None):
    """n science frames (FLUX, CENTER, CORO..., CENTER) of one pointing."""
    types = ["OBJECT,FLUX", "OBJECT,CENTER"] + ["OBJECT"] * (n - 3) + ["OBJECT,CENTER"]
    targ = targ if isinstance(targ, list) else [targ or (0.0, 0.0)] * n
    rows = []
    for i, dpr in enumerate(types):
        rows.append({
            "OBJECT": obj, "RA": ra + i * drift_arcsec_per_frame * ARCSEC / np.cos(np.deg2rad(dec)), "DEC": dec,
            "MJD_OBS": start + i * 2 / 1440, "DATE_OBS": f"{i:04d}", "NIGHT_START": NIGHT, "DPR_TYPE": dpr,
            "TARG_ALPHA": targ[i][0], "TARG_DELTA": targ[i][1],
        })
    return rows


@pytest.fixture
def build(monkeypatch):
    """Run create_observation_table with the metadata helpers stubbed to the essentials.

    The real metadata code needs full SPHERE headers; vetting only needs the sequence
    grouping, so the helpers are replaced by stand-ins that keep the frame count.
    """
    from spherical.database import observation_table as ot

    def science(table_of_files, *args, **kwargs):
        return None, None, None, None, table_of_files

    monkeypatch.setattr(ot, "filter_for_science_frames", science)
    monkeypatch.setattr(ot, "select_primary_science_frames",
                        lambda group, ndit_key: ("CORO", group[group["DPR_TYPE"] == "OBJECT"], {"TOTAL_EXPTIME_CORO": 1.0}))
    monkeypatch.setattr(ot, "evaluate_observation_flags", lambda group, ndit_key: {"FLUX_FLAG": False})
    monkeypatch.setattr(ot, "calculate_observation_metadata",
                        lambda observation_files, **kwargs: {"NCUBES": len(observation_files),
                                                             "OBS_START": float(np.min(observation_files["MJD_OBS"]))})
    monkeypatch.setattr(ot, "compute_hci_ready", lambda metadata, polarimetry: True)

    def run(file_rows, target_rows, overrides=()):
        files = Table(rows=file_rows)
        files["DB_FILTER"] = "DB_H23"
        files["NDIT"] = 1
        targets = Table(rows=target_rows)
        targets["DISTANCE"] = 10.0  # a stellar-target column the other table must not carry
        return create_observation_table(files, targets, instrument="irdis", remove_fillers=False,
                                        overrides=list(overrides))

    return run


def _target(main_id, ra, dec, **extra):
    # Every row has the same keys: Table(rows=...) rejects dicts with different keys.
    row = {"MAIN_ID": main_id, "RA_HEADER": ra, "DEC_HEADER": dec, "RA_DEG": ra, "DEC_DEG": dec,
           "PMRA": 0.0, "PMDEC": 0.0, "POS_DIFF": 1.0, "ID_HD": "", "ID_HIP": "", "OBJ_HEADER": main_id}
    row.update(extra)
    return row


def test_returns_three_tables_and_unchanged_targets(build):
    targets = [_target("HD 1160", 10.0, -30.0, ID_HIP="HIP 1272")]
    obs, tgt, other = build(_file_rows("HIP_1272", 10.0, -30.0), targets)
    assert len(obs) == 1 and len(tgt) == 1 and len(other) == 0
    assert obs["FIELD_TARGETS"][0] == "" and obs["VETTING_FLAG"][0] == ""
    assert list(other.colnames[: len(OTHER_LEADING_COLUMNS)]) == OTHER_LEADING_COLUMNS
    assert "_FILE_INDEX" not in obs.colnames


def test_shared_sequence_goes_to_named_owner(build):
    targets = [_target("HD  1160", 10.0, -30.0, ID_HIP="HIP 1272"),
               _target("HD  1160C", 10.0, -30.0 + 5 * ARCSEC)]
    obs, tgt, other = build(_file_rows("HIP_1272", 10.0, -30.0 + 3 * ARCSEC), targets)
    assert obs["MAIN_ID"].tolist() == ["HD  1160"]
    assert obs["FIELD_TARGETS"][0] == "HD  1160C"
    assert len(tgt) == 2  # the target table is not pruned


def test_moving_sequence_goes_to_other_table(build):
    targets = [_target("2MASS J18234340-2345536", 276.0, -23.0, POS_DIFF=107.0, OBJ_HEADER="476 Hedwig")]
    rows = _file_rows("476 Hedwig", 276.0, -23.0, n=8, drift_arcsec_per_frame=0.6)
    obs, _, other = build(rows, targets)
    assert len(obs) == 0
    assert other["CATEGORY"].tolist() == ["solar_system"]
    assert other["MATCHED_MAIN_ID"][0] == "2MASS J18234340-2345536"
    assert other["OBJECT"][0] == "476 Hedwig"
    assert "DISTANCE" not in other.colnames  # stellar-target columns are dropped


def test_override_and_offset_categories(build):
    targets = [_target("BD-00 413", 40.67, -0.01, POS_DIFF=86.9),
               _target("TYC 1", 100.0, -50.0, POS_DIFF=49.8)]
    rows = _file_rows("NGC1068", 40.67, -0.01) + _file_rows("S CrA", 100.0, -50.0)
    overrides = [mv.Override(mv.re.compile("ngc\\s*1068", mv.re.IGNORECASE), "non_stellar", "NGC 1068")]
    obs, _, other = build(rows, targets, overrides)
    assert len(obs) == 0
    assert sorted(other["CATEGORY"].tolist()) == ["non_stellar", "unmatched"]


def test_target_change_flags_and_clears_hci_ready(build):
    targ = [(40003.844, -290216.40)] * 3 + [(40003.956, -290227.85)] * 3
    targets = [_target("HD  25284", 60.0, -29.0, ID_HD="HD 25284")]
    obs, _, _ = build(_file_rows("HD_25284", 60.0, -29.0, targ=targ), targets)
    assert obs["VETTING_FLAG"][0] == "target_changed"
    assert not obs["HCI_READY"][0]


def test_ambiguous_owner_flags_and_clears_hci_ready(build):
    targets = [_target("CD-30 1", 10.0, -30.0), _target("CD-30 2", 10.0, -30.0 + 9.3 * ARCSEC)]
    obs, _, _ = build(_file_rows("No name", 10.0, -30.0 + 4.65 * ARCSEC), targets)
    assert len(obs) == 1
    assert obs["VETTING_FLAG"][0] == "ambiguous_owner"
    assert not obs["HCI_READY"][0]


def test_rebuild_from_written_tables_is_identical(build, tmp_path):
    targets = [_target("HD  1160", 10.0, -30.0, ID_HIP="HIP 1272"),
               _target("HD  1160C", 10.0, -30.0 + 5 * ARCSEC),
               _target("2MASS J1", 276.0, -23.0, POS_DIFF=107.0)]
    rows = _file_rows("HIP_1272", 10.0, -30.0 + 3 * ARCSEC) + _file_rows("476 Hedwig", 276.0, -23.0, n=8,
                                                                           drift_arcsec_per_frame=0.6)
    obs1, tgt1, other1 = build(rows, targets)
    tgt1.write(tmp_path / "t.fits")
    obs2, _, other2 = build(rows, [dict(zip(tgt1.colnames, r)) for r in Table.read(tmp_path / "t.fits")])
    assert obs1["MAIN_ID"].tolist() == obs2["MAIN_ID"].tolist()
    assert obs1["FIELD_TARGETS"].tolist() == obs2["FIELD_TARGETS"].tolist()
    assert other1["CATEGORY"].tolist() == other2["CATEGORY"].tolist()


def test_no_files_gives_empty_but_valid_tables(build, tmp_path):
    targets = [_target("HD 1", 10.0, -30.0)]
    obs, tgt, other = build(_file_rows("far", 50.0, 10.0), targets)
    assert len(other) == 0
    other.write(tmp_path / "other.fits")  # an empty other table must still be writable
    assert Table.read(tmp_path / "other.fits").colnames[: len(OTHER_LEADING_COLUMNS)] == OTHER_LEADING_COLUMNS


def test_high_proper_motion_star_with_different_ob_epochs_is_not_flagged(build):
    # Sirius 2014-12-06: two OBs give coordinates 9.6" apart for the same star.
    targ = [(64508.917, -164258.017)] * 3 + [(64509.565, -164255.857)] * 3
    targets = [_target("* alf CMa", 101.29, -16.72, PMRA=-546.0, PMDEC=-1223.0, ID_HD="HD 48915")]
    obs, _, _ = build(_file_rows("HD 48915", 101.29, -16.72, targ=targ), targets)
    assert obs["VETTING_FLAG"][0] == ""

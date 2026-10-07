"""Unit tests for spherical.database.match_vetting (#224)."""

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from spherical.database import match_vetting as mv


def _target(**columns):
    defaults = {"MAIN_ID": "HD 1160", "ID_HD": "HD 1160", "ID_HIP": "HIP 1272", "ID_TYC": "",
                "ID_2MASS": "2MASS J00155605+0447405", "ID_GAIA_DR3": "Gaia DR3 2552928187080872832"}
    defaults.update(columns)
    return Table({key: [value] for key, value in defaults.items()})[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*  51 Eri", "51eri"),
        ("51_ERI", "51eri"),
        ("V* UX Tau A", "uxtaua"),
        ("HDE 305524", "hd305524"),
        ("HIP_1272", "hip1272"),
        (b"HD  1160C", "hd1160c"),
        ("", ""),
        ("nan", ""),
        ("--", ""),
    ],
)
def test_normalize_designation(raw, expected):
    assert mv.normalize_designation(raw) == expected


def test_target_identifiers_split_pipes_and_skip_empty():
    target = _target(ID_HD="HD 135344|HD 135344A", ID_TYC="")
    ids = mv.target_identifiers(target)
    assert {"hd1160", "hd135344", "hd135344a", "hip1272"} <= ids
    assert "" not in ids


def test_target_identifiers_tolerate_masked_and_missing_columns():
    table = Table({"MAIN_ID": ["HD 1160"], "ID_HIP": MaskedColumn(["HIP 1272"], mask=[True])})
    assert mv.target_identifiers(table[0]) == {"hd1160"}


def test_exact_match_only():
    ids = mv.target_identifiers(_target(MAIN_ID="HD  25284B", ID_HD="HD 25284B"))
    assert mv.normalize_designation("HD_25284") not in ids


def test_as_float_handles_masked_and_text():
    assert np.isnan(mv.as_float(np.ma.masked))
    assert np.isnan(mv.as_float("abc"))
    assert mv.as_float("12.5") == 12.5


def _track(steps_arcsec, minutes=2.0, ra0=150.0, dec0=-30.0):
    """Header positions from successive (dx, dy) steps in arcsec, one frame every `minutes`."""
    x = np.concatenate([[0.0], np.cumsum([s[0] for s in steps_arcsec])])
    y = np.concatenate([[0.0], np.cumsum([s[1] for s in steps_arcsec])])
    ra = ra0 + x / 3600.0 / np.cos(np.deg2rad(dec0))
    dec = dec0 + y / 3600.0
    mjd = 60000.0 + np.arange(len(x)) * minutes / 1440.0
    return ra, dec, mjd


def test_steady_mover_is_caught():
    ra, dec, mjd = _track([(0.5, 0.2)] * 10)  # 0.54"/step every 2 min
    assert mv.steady_drift_rate(ra, dec, mjd) == pytest.approx(np.hypot(0.5, 0.2) / 2.0, rel=1e-3)


def test_slow_steady_mover_above_threshold():
    ra, dec, mjd = _track([(0.08, 0.0)] * 10, minutes=2.0)  # 0.04"/min
    assert mv.steady_drift_rate(ra, dec, mjd) > mv.DRIFT_RATE_MIN


@pytest.mark.parametrize(
    "steps",
    [
        [(9.3, 0.0), (-9.3, 0.0)] * 5,         # 55 Eri: toggling between components
        [(0.0, 0.0), (5.0, 0.0)],                # single 5" offset in a 3-frame sequence
        [(833.0, 0.0), (-833.0, 0.0)] * 3,       # two targets alternated under one name
        [(0.0, 0.0)] * 10,                       # frozen header coordinates
        [(0.0, 0.0)] * 7 + [(3.0, 0.0)] * 2,     # re-acquisition jumps in a sidereal sequence
    ],
)
def test_repointings_are_not_motion(steps):
    ra, dec, mjd = _track(steps)
    assert mv.steady_drift_rate(ra, dec, mjd) == 0.0


def test_two_frames_are_not_enough():
    ra, dec, mjd = _track([(0.5, 0.0)])
    assert mv.steady_drift_rate(ra, dec, mjd) == 0.0


def test_unsorted_input_is_sorted_by_time():
    ra, dec, mjd = _track([(0.5, 0.0)] * 6)
    order = np.array([3, 0, 6, 1, 5, 2, 4])
    assert mv.steady_drift_rate(ra[order], dec[order], mjd[order]) > mv.DRIFT_RATE_MIN


def test_moving_sequences_keys_by_object_and_night():
    ra_m, dec_m, mjd_m = _track([(0.5, 0.0)] * 5)
    ra_s, dec_s, mjd_s = _track([(0.0, 0.0)] * 5, ra0=10.0)
    files = Table({
        "OBJECT": ["476 Hedwig"] * 6 + ["HD 1160"] * 6,
        "NIGHT_START": ["2017-05-27"] * 12,
        "RA": np.concatenate([ra_m, ra_s]),
        "DEC": np.concatenate([dec_m, dec_s]),
        "MJD_OBS": np.concatenate([mjd_m, mjd_s]),
    })
    moving = mv.moving_sequences(files)
    assert set(moving) == {("476 Hedwig", "2017-05-27")}


def test_moving_sequences_of_empty_table():
    empty = Table(names=["OBJECT", "NIGHT_START", "RA", "DEC", "MJD_OBS"], dtype=[str, str, float, float, float])
    assert mv.moving_sequences(empty) == {}


OVERRIDES = [
    mv.Override(mv.re.compile(r"chariklo", mv.re.IGNORECASE), "solar_system", "Chariklo"),
    mv.Override(mv.re.compile(r"ngc\s*1068", mv.re.IGNORECASE), "non_stellar", "NGC 1068"),
    mv.Override(mv.re.compile(r"^\s*hd\s*216127", mv.re.IGNORECASE), "keep", "re-pointing, not motion"),
]
IDS = {"hd1160", "hip1272"}


def test_packaged_overrides_load_and_cover_seed_cases():
    overrides = mv.load_overrides()
    assert mv.match_override(["Chariklo moving object"], overrides).category == "solar_system"
    assert mv.match_override(["NGC1068 nucleus"], overrides).category == "non_stellar"
    assert mv.match_override(["WD2226-210"], overrides).category == "non_stellar"
    assert mv.match_override(["HD 1160"], overrides) is None


def test_load_overrides_rejects_unknown_category(tmp_path):
    path = tmp_path / "o.csv"
    path.write_text("pattern,category,reason\nfoo,galaxy,typo\n")
    with pytest.raises(ValueError, match="galaxy"):
        mv.load_overrides(path)


def test_override_wins_over_everything():
    moving = {("Chariklo moving object", "2016-07-20"): 0.5}
    assert mv.classify_sequence(["Chariklo moving object"], "2016-07-20", 30.0, set(), moving, OVERRIDES) == (
        "solar_system", "override: Chariklo")
    assert mv.classify_sequence(["NGC1068"], "2019-10-29", 86.9, set(), {}, OVERRIDES)[0] == "non_stellar"


def test_keep_exempts_from_drift_and_offset():
    moving = {("HD 216127", "2021-10-04"): 0.066}
    assert mv.classify_sequence(["HD 216127"], "2021-10-04", 40.0, set(), moving, OVERRIDES) is None


def test_drift_makes_solar_system():
    moving = {("476 Hedwig", "2017-05-27"): 0.266}
    category, reason = mv.classify_sequence(["476 Hedwig"], "2017-05-27", 106.9, set(), moving, OVERRIDES)
    assert category == "solar_system"
    assert reason == 'drift 0.27"/min'


def test_unmatched_needs_offset_and_no_name():
    assert mv.classify_sequence(["S CrA"], "2015-05-01", 49.8, IDS, {}, OVERRIDES) == (
        "unmatched", 'offset 50" no name match')
    assert mv.classify_sequence(["HIP_1272"], "2015-05-01", 49.8, IDS, {}, OVERRIDES) is None
    assert mv.classify_sequence(["S CrA"], "2015-05-01", 5.1, IDS, {}, OVERRIDES) is None


def test_missing_offset_never_unmatched():
    assert mv.classify_sequence(["S CrA"], "2015-05-01", np.nan, IDS, {}, OVERRIDES) is None


def test_target_coordinate_spread_from_ob_coordinates():
    files = Table({"TARG_ALPHA": [40003.844, 40003.844, 40003.956], "TARG_DELTA": [-290216.40, -290216.40, -290227.85]})
    assert mv.target_coordinate_spread(files) == pytest.approx(11.5, abs=0.3)


def test_target_coordinate_spread_without_columns_or_frames():
    assert mv.target_coordinate_spread(Table({"RA": [1.0, 2.0]})) == 0.0
    assert mv.target_coordinate_spread(Table({"TARG_ALPHA": [10000.0], "TARG_DELTA": [-10000.0]})) == 0.0


def test_target_coordinate_spread_ignores_masked_values():
    files = Table({"TARG_ALPHA": MaskedColumn([40003.844, 0.0, 40003.844], mask=[False, True, False]),
                   "TARG_DELTA": MaskedColumn([-290216.40, 0.0, -290216.40], mask=[False, True, False])})
    assert mv.target_coordinate_spread(files) == 0.0


def test_format_field_targets_caps_the_list():
    assert mv.format_field_targets([]) == ""
    assert mv.format_field_targets(["HD  1160C", "HD  1160C"]) == "HD  1160C"
    names = [f"NGC 3603 {i}" for i in range(8)]
    assert mv.format_field_targets(names) == "|".join(sorted(names)[:5]) + "|+3 more"


def test_vetting_parameters_record_the_constants():
    params = mv.vetting_parameters()
    assert params["drift_rate_min"] == mv.DRIFT_RATE_MIN
    assert params["max_match_offset"] == mv.MAX_MATCH_OFFSET


import astropy.units as u  # noqa: E402

CONE = 15 * u.arcsec
ARCSEC = 1 / 3600.0


def _targets(rows):
    """rows: (MAIN_ID, header RA, header DEC, SIMBAD RA, SIMBAD DEC) with zero proper motion."""
    return Table({
        "MAIN_ID": [r[0] for r in rows],
        "RA_HEADER": [r[1] for r in rows], "DEC_HEADER": [r[2] for r in rows],
        "RA_DEG": [r[3] for r in rows], "DEC_DEG": [r[4] for r in rows],
        "PMRA": [0.0] * len(rows), "PMDEC": [0.0] * len(rows),
    })


def _files(rows):
    """rows: (OBJECT, RA, DEC)."""
    return Table({
        "OBJECT": [r[0] for r in rows], "RA": [float(r[1]) for r in rows], "DEC": [float(r[2]) for r in rows],
        "MJD_OBS": [60000.0] * len(rows),
    }, dtype=[str, float, float, float])


def _ids(targets):
    return [mv.target_identifiers(t) for t in targets]


def test_single_candidate_owns_as_before():
    targets = _targets([("HD 1", 10.0, -30.0, 10.0, -30.0), ("HD 2", 20.0, -30.0, 20.0, -30.0)])
    files = _files([("HD_1", 10.0, -30.0), ("HD_2", 20.0, -30.0), ("far", 30.0, -30.0)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [0, 1, -1]
    assert not own.ambiguous.any()
    assert own.others == [(), (), ()]
    assert own.has_candidate_files.tolist() == [True, True]


def test_name_match_wins_over_distance():
    # HD 1160 and HD 1160C 5" apart; the file points nearer C but is named after HD 1160.
    a = ("HD 1160", 10.0, -30.0, 10.0, -30.0)
    c = ("HD 1160C", 10.0, -30.0 + 5 * ARCSEC, 10.0, -30.0 + 5 * ARCSEC)
    targets = _targets([a, c])
    targets["ID_HIP"] = ["HIP 1272", ""]
    files = _files([("HIP_1272", 10.0, -30.0 + 4 * ARCSEC)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [0]
    assert own.others == [(1,)]
    assert not own.ambiguous[0]


def test_nearest_simbad_position_without_name_match():
    a = ("CD-30 1", 10.0, -30.0, 10.0, -30.0)
    b = ("CD-30 2", 10.0, -30.0 + 8 * ARCSEC, 10.0, -30.0 + 8 * ARCSEC)
    targets = _targets([a, b])
    files = _files([("No name", 10.0, -30.0 + 6 * ARCSEC)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [1]
    assert not own.ambiguous[0]


def test_near_tie_is_ambiguous():
    a = ("CD-30 1", 10.0, -30.0, 10.0, -30.0)
    b = ("CD-30 2", 10.0, -30.0 + 9.3 * ARCSEC, 10.0, -30.0 + 9.3 * ARCSEC)
    targets = _targets([a, b])
    files = _files([("No name", 10.0, -30.0 + 4.4 * ARCSEC)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.ambiguous[0]


def test_never_owned_by_a_target_whose_cone_misses_it():
    # B's SIMBAD position is closest, but its header cone (20" away) does not contain the file.
    a = ("CD-30 1", 10.0, -30.0, 10.0, -30.0 + 10 * ARCSEC)
    b = ("CD-30 2", 10.0, -30.0 + 20 * ARCSEC, 10.0, -30.0)
    targets = _targets([a, b])
    files = _files([("No name", 10.0, -30.0)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [0]


def test_masked_proper_motion_and_position_fall_back():
    a = ("CD-30 1", 10.0, -30.0, 10.0, -30.0)
    b = ("CD-30 2", 10.0, -30.0 + 8 * ARCSEC, 10.0, -30.0 + 8 * ARCSEC)
    targets = _targets([a, b])
    targets["PMRA"] = MaskedColumn([0.0, 0.0], mask=[True, True])
    targets["RA_DEG"] = MaskedColumn([10.0, 10.0], mask=[False, True])
    files = _files([("No name", 10.0, -30.0 + 6 * ARCSEC)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [1]


def test_empty_inputs():
    targets = _targets([("HD 1", 10.0, -30.0, 10.0, -30.0)])
    own = mv.assign_file_owners(_files([]), targets, CONE, _ids(targets))
    assert len(own.owner) == 0 and own.has_candidate_files.tolist() == [False]
    own = mv.assign_file_owners(_files([("x", 10.0, -30.0)]), targets[:0], CONE, [])
    assert own.owner.tolist() == [-1]


def test_slow_pointing_creep_is_not_motion():
    # TW Hya, IFS SAM 2018-04-29: three frames 30 min apart creeping 1.1" in a straight line.
    ra, dec, mjd = _track([(0.18, 0.07), (0.78, 0.51)], minutes=30.0)
    assert mv.steady_drift_rate(ra, dec, mjd) <= mv.DRIFT_RATE_MIN


def test_slowest_known_mover_stays_above_threshold():
    # 121 Hermione, IRDIS 2015-09-10, 0.049"/min.
    ra, dec, mjd = _track([(0.098, 0.0)] * 3, minutes=2.0)
    assert mv.steady_drift_rate(ra, dec, mjd) > mv.DRIFT_RATE_MIN


def test_target_change_limit_allows_for_proper_motion():
    # Sirius: OB coordinates at different epochs differ by 9.6" for the same star.
    assert mv.target_change_limit(-546.0, -1223.0) > 9.6
    assert mv.target_change_limit(20.0, -30.0) == mv.TARGET_CHANGE_SEP
    assert mv.target_change_limit(np.nan, np.nan) == mv.TARGET_CHANGE_SEP


def test_packaged_overrides_cover_short_asteroid_sequences():
    overrides = mv.load_overrides()
    assert mv.match_override(["1999KW4"], overrides).category == "solar_system"
    assert mv.match_override(["71_Niobe"], overrides).category == "solar_system"


def test_ownership_needs_no_scipy(monkeypatch):
    """The database install has no scipy; astropy's search_around_sky needs it for its KD-tree."""
    import sys

    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    targets = _targets([("HD 1", 10.0, -30.0, 10.0, -30.0), ("HD 2", 20.0, -30.0, 20.0, -30.0)])
    files = _files([("HD_1", 10.0, -30.0), ("HD_2", 20.0, -30.0 + 14 * ARCSEC), ("far", 30.0, -30.0)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [0, 1, -1]


def test_non_finite_coordinates_are_skipped_not_fatal():
    targets = _targets([("HD 1", 10.0, -30.0, 10.0, -30.0), ("HD 2", np.nan, np.nan, np.nan, np.nan)])
    targets["RA_HEADER"] = MaskedColumn([10.0, 20.0], mask=[False, True])
    files = _files([("HD_1", 10.0, -30.0), ("bad", np.nan, -30.0)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [0, -1]
    assert own.has_candidate_files.tolist() == [True, False]


def test_cone_boundary_is_strict_and_wraps_in_ra():
    targets = _targets([("HD 1", 0.001, 10.0, 0.001, 10.0)])
    just_inside = 0.001 - 14.9 * ARCSEC / np.cos(np.deg2rad(10.0))  # across RA 0/360
    files = _files([("a", just_inside % 360.0, 10.0), ("b", 0.001, 10.0 + 15.1 * ARCSEC)])
    own = mv.assign_file_owners(files, targets, CONE, _ids(targets))
    assert own.owner.tolist() == [0, -1]

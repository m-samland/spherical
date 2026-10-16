"""``ROTATION`` when the first or last file of a sequence has a wrong derotator position (#186).

``compute_angles`` takes the right ascension of every frame from its file's ``INS4 DROT2 RA``
and the declination of the first frame for all. A file whose derotator position is wrong,
typically the last one of a sequence that was cut short (its ``TEL PARANG END`` is 0 and
its AO loops are often open), then gets a wrong parallactic angle. When that file is the
first or the last, ``ROTATION`` (last minus first) is wrong by tens of degrees.

The fixture holds the CORO files of three real IRDIS sequences from the v3.0.0 file table.
HD 169142 and HIP 116063 have a last file with a wrong derotator position; HD 58526 has a
last file with an unset ``TEL PARANG END`` but a correct derotator position, and must not change.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from spherical.database.observation_table import compute_derotation_info, replace_end_derotator_outliers

FIXTURE = Path(__file__).parent / "data" / "rotation_end_outliers.csv"


def _sequence(object_name):
    table = Table.read(FIXTURE, format="csv")
    return table[np.asarray(table["OBJECT"]) == object_name]


def _rotation(table):
    return compute_derotation_info(table, polarimetry=False)["ROTATION"]


def test_last_file_with_wrong_derotator_position_no_longer_breaks_rotation():
    # ESO's TEL PARANG: -84.77 at the start of the first file, -70.55 at the start of the
    # last, 1.9 deg per file, so about 16.4 deg.
    assert _rotation(_sequence("HD_169142")) == pytest.approx(16.4, abs=3.0)


def test_rotation_of_a_longer_sequence_with_the_same_defect():
    # -27.00 at the start of the first file to 4.94 at the start of the last, plus one file.
    assert _rotation(_sequence("HIP_116063")) == pytest.approx(34.1, abs=3.0)


def test_unset_parang_end_with_a_correct_derotator_position_changes_nothing():
    table = _sequence("HD 58526")
    assert replace_end_derotator_outliers(table) is table
    assert _rotation(table) == pytest.approx(3.7, abs=0.3)


def _files(ra, dec):
    return Table({"DROT2_RA": np.array(ra, dtype=float), "DROT2_DEC": np.array(dec, dtype=float)})


GOOD_RA, GOOD_DEC = 120000.0, -300000.0  # 12h00m00s, -30d00m00s
WRONG_RA, WRONG_DEC = 123000.0, -250000.0


def test_wrong_last_file_takes_the_position_of_the_others():
    files = _files([GOOD_RA, GOOD_RA + 0.01, GOOD_RA, WRONG_RA], [GOOD_DEC, GOOD_DEC, GOOD_DEC, WRONG_DEC])
    fixed = replace_end_derotator_outliers(files)
    assert fixed["DROT2_RA"][3] == GOOD_RA and fixed["DROT2_DEC"][3] == GOOD_DEC
    assert files["DROT2_RA"][3] == WRONG_RA  # the input is not modified


def test_wrong_first_file_is_replaced_too():
    files = _files([WRONG_RA, GOOD_RA, GOOD_RA, GOOD_RA], [WRONG_DEC, GOOD_DEC, GOOD_DEC, GOOD_DEC])
    fixed = replace_end_derotator_outliers(files)
    assert fixed["DROT2_RA"][0] == GOOD_RA and fixed["DROT2_DEC"][0] == GOOD_DEC


def test_both_ends_wrong_and_the_middle_consistent():
    files = _files([WRONG_RA, GOOD_RA, GOOD_RA, GOOD_RA, WRONG_RA], [WRONG_DEC, GOOD_DEC, GOOD_DEC, GOOD_DEC, WRONG_DEC])
    fixed = replace_end_derotator_outliers(files)
    assert list(fixed["DROT2_RA"]) == [GOOD_RA] * 5


def test_a_wrong_file_in_the_middle_is_left_alone():
    files = _files([GOOD_RA, GOOD_RA, WRONG_RA, GOOD_RA], [GOOD_DEC, GOOD_DEC, WRONG_DEC, GOOD_DEC])
    assert replace_end_derotator_outliers(files) is files


def test_two_files_cannot_tell_which_is_wrong():
    files = _files([GOOD_RA, WRONG_RA], [GOOD_DEC, WRONG_DEC])
    assert replace_end_derotator_outliers(files) is files


def test_a_steadily_moving_target_is_left_alone():
    # 30 arcsec of right ascension per file: nothing is an outlier among the others.
    ra = [GOOD_RA + 2.0 * i for i in range(8)]
    files = _files(ra, [GOOD_DEC] * 8)
    assert replace_end_derotator_outliers(files) is files

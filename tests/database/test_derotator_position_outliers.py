"""Files whose derotator position (``INS4 DROT2 RA/DEC``) disagrees with the rest of the sequence (#186).

``compute_angles`` takes every frame's hour angle from its file's ``INS4 DROT2 RA``. In a few
files, typically the last one of a sequence cut short, the end-of-exposure keywords were
written after the telescope had started to move away: ``TEL PARANG END`` is 0 and the
derotator position is that of another point on the sky, although the start keywords show the
star was still tracked. The field rotation of such a file follows the star like the others, so
the file takes the sequence's position, keeps its own times, and its integrations are flagged.

The fixture holds the CORO files of three real IRDIS sequences from the v3.0.0 file table.
HD 169142 and HIP 116063 have a last file with a wrong derotator position; HD 58526 has a
last file with an unset ``TEL PARANG END`` but a correct derotator position.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from spherical.database import metadata
from spherical.database.observation_table import compute_derotation_info

FIXTURE = Path(__file__).parent / "data" / "derotator_position_outliers.csv"
FLAG = "DROT2 POSITION OUTLIER"


def _sequence(object_name):
    table = Table.read(FIXTURE, format="csv")
    return table[np.asarray(table["OBJECT"]) == object_name]


def _frames(object_name):
    frames = metadata.prepare_dataframe(_sequence(object_name))
    metadata.compute_times(frames)
    metadata.compute_angles(frames)
    return frames


def _rotation(object_name):
    return compute_derotation_info(_sequence(object_name), polarimetry=False)["ROTATION"]


def test_rotation_with_a_wrong_last_file():
    # ESO's TEL PARANG: -84.77 at the start of the first file, -70.55 at the start of the
    # last, 1.9 deg per file, so about 16.4 deg.
    assert _rotation("HD_169142") == pytest.approx(16.4, abs=3.0)


def test_rotation_of_a_longer_sequence_with_the_same_defect():
    # -27.00 at the start of the first file to 4.94 at the start of the last, plus one file.
    assert _rotation("HIP_116063") == pytest.approx(34.1, abs=3.0)


def test_the_wrong_file_is_flagged_and_its_angle_follows_the_star():
    frames = _frames("HD_169142")
    starts = np.flatnonzero(frames["DIT INDEX"].to_numpy() == 0)
    last_file = np.arange(len(frames)) >= starts[-1]
    assert frames[FLAG].to_numpy().tolist() == last_file.tolist()
    # Its first integration matches ESO's TEL PARANG START of that file (-70.551 deg).
    assert frames["PARANG START"].to_numpy()[starts[-1]] == pytest.approx(-70.551, abs=0.5)


def test_unset_parang_end_with_a_correct_derotator_position_changes_nothing():
    frames = _frames("HD 58526")
    assert not frames[FLAG].any()
    assert _rotation("HD 58526") == pytest.approx(3.7, abs=0.3)


GOOD_RA, GOOD_DEC = 120000.0, -300000.0  # 12h00m00s, -30d00m00s
WRONG_RA, WRONG_DEC = 123000.0, -250000.0


def _rows(positions, dits=2):
    """Per-integration arrays for files at ``positions``, ``dits`` integrations each."""
    ra = np.repeat([p[0] for p in positions], dits).astype(float)
    dec = np.repeat([p[1] for p in positions], dits).astype(float)
    return ra, dec, np.repeat(np.arange(len(positions)), dits)


GOOD, WRONG = (GOOD_RA, GOOD_DEC), (WRONG_RA, WRONG_DEC)


@pytest.mark.parametrize("wrong_file", [0, 1, 3])
def test_a_wrong_file_takes_the_position_of_the_others(wrong_file):
    positions = [GOOD, (GOOD_RA + 0.01, GOOD_DEC), GOOD, GOOD]
    positions[wrong_file] = WRONG
    ra, dec, file_id = _rows(positions)
    new_ra, new_dec, outlier = metadata.robust_derotator_position(ra, dec, file_id)
    assert outlier.tolist() == (file_id == wrong_file).tolist()
    assert np.all(new_ra[outlier] == GOOD_RA) and np.all(new_dec[outlier] == GOOD_DEC)
    assert np.all(new_ra[~outlier] == ra[~outlier])
    assert ra[file_id == wrong_file][0] == WRONG_RA  # the input is not modified


def test_two_files_cannot_tell_which_is_wrong():
    ra, dec, file_id = _rows([GOOD, WRONG])
    _, _, outlier = metadata.robust_derotator_position(ra, dec, file_id)
    assert not outlier.any()


def test_a_steadily_moving_target_is_left_alone():
    # 2 s of right ascension (26 arcsec) per file: the others are not consistent with each other.
    ra, dec, file_id = _rows([(GOOD_RA + 2.0 * i, GOOD_DEC) for i in range(8)])
    _, _, outlier = metadata.robust_derotator_position(ra, dec, file_id)
    assert not outlier.any()


def test_half_the_files_wrong_is_left_alone():
    ra, dec, file_id = _rows([GOOD, GOOD, WRONG, WRONG])
    _, _, outlier = metadata.robust_derotator_position(ra, dec, file_id)
    assert not outlier.any()

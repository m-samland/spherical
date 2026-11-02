import pytest

from spherical.conditions.ambient import parse_ambient_csv, query_ambient
from spherical.conditions.archive import Deadline, find_sparta_files

pytestmark = pytest.mark.remote_data

EXPECTED = [
    "SPHER.2017-09-28T07:50:56.740",
    "SPHER.2017-09-28T08:18:45.314",
    "SPHER.2017-09-28T08:32:37.648",
    "SPHER.2017-09-28T08:46:26.164",
    "SPHER.2017-09-28T09:00:15.206",
    "SPHER.2017-09-28T09:14:03.195",
    "SPHER.2017-09-28T09:27:54.026",
    "SPHER.2017-09-28T09:31:29.983",
    "SPHER.2017-09-28T09:35:55.352",
]


def test_51eri_2017_sparta_and_dimm():
    files = find_sparta_files([1837295], 58024.30, 58024.41, deadline=Deadline(60))
    assert list(files["dp_id"]) == EXPECTED
    lhatpro = parse_ambient_csv(query_ambient("lhatpro", 58024.319, 58024.403, deadline=Deadline(60)))
    assert len(lhatpro) > 30

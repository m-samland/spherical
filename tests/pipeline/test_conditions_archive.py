import urllib.error

import pytest
from astropy.time import Time

from spherical.conditions import ambient
from spherical.conditions.archive import Deadline, SourceFailed, download_sparta_files, sparta_query

DIMM = """# ESO ambient
# DIMM Paranal

Date time,DIMM Seeing ["],Relative Flux RMS,Relative Flux RMS base time [s]
2017-09-28T07:41:05,0.349,0.007175,546
2017-09-28T07:41:35,0.351,0.007001,546
"""


def test_query_uses_all_ob_ids_and_mjd_window():
    query = sparta_query([1837295, 1837296], 58024.30, 58024.41)
    assert "ob_id IN (1837295, 1837296)" in query
    assert "mjd_obs BETWEEN 58024.3 AND 58024.41" in query
    assert "dp_type='OBJECT,AO'" in query
    loose = sparta_query([1837295], 58024.30, 58024.41, filtered=False)
    assert "ob_id" not in loose.split("WHERE")[1]
    assert "dp_type" not in loose.split("WHERE")[1]


def test_deadline_expired_raises_before_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("urlopen called")

    monkeypatch.setattr(ambient.urllib.request, "urlopen", fail)
    with pytest.raises(SourceFailed):
        ambient.query_ambient("dimm", 58024.3, 58024.4, deadline=Deadline(0))


def test_http_error_becomes_source_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(ambient.urllib.request, "urlopen", boom)
    with pytest.raises(SourceFailed):
        ambient.query_ambient("dimm", 58024.3, 58024.4, deadline=Deadline(60))


def test_parse_ambient_csv():
    frame = ambient.parse_ambient_csv(DIMM)
    assert len(frame) == 2
    assert frame["mjd"].iloc[0] == Time("2017-09-28T07:41:05").mjd
    assert ambient.parse_ambient_csv("# a\n# b\n").empty


def test_download_skips_present_files(tmp_path, monkeypatch):
    (tmp_path / "SPHER.X.fits.Z").write_bytes(b"")

    def fail(*args, **kwargs):
        raise AssertionError("retrieve_data called")

    monkeypatch.setattr("astroquery.eso.core.EsoClass.retrieve_data", fail)
    paths = download_sparta_files(["SPHER.X"], tmp_path, deadline=Deadline(60))
    assert paths == [tmp_path / "SPHER.X.fits.Z"]


class _Reply:
    def __init__(self, text):
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._text.encode()


def test_tap_reply_is_parsed_and_timeout_is_bounded(monkeypatch):
    from spherical.conditions import archive

    seen = {}

    def fake(url, timeout):
        seen["timeout"] = timeout
        seen["url"] = url
        return _Reply('dp_id,mjd_obs,ob_id,dp_type\nSPHER.B,58024.4,1,"OBJECT,AO"\nSPHER.A,58024.3,1,"OBJECT,AO"\nSPHER.C,58024.3,1,"OBJECT,SKY"\n')

    monkeypatch.setattr(archive.urllib.request, "urlopen", fake)
    frame = archive.find_sparta_files([1], 58024.3, 58024.41, deadline=Deadline(5))
    assert list(frame["dp_id"]) == ["SPHER.A", "SPHER.B"]
    assert seen["timeout"] <= 5
    assert "tap_obs/sync" in seen["url"]


@pytest.mark.parametrize("error", [urllib.error.URLError("down"), TimeoutError("slow")])
def test_tap_errors_become_source_failed(monkeypatch, error):
    from spherical.conditions import archive

    def boom(url, timeout):
        raise error

    monkeypatch.setattr(archive.urllib.request, "urlopen", boom)
    with pytest.raises(SourceFailed):
        archive.find_sparta_files([1], 58024.3, 58024.41, deadline=Deadline(5))


def test_tap_expired_deadline_raises_before_network(monkeypatch):
    from spherical.conditions import archive

    monkeypatch.setattr(archive.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    with pytest.raises(SourceFailed):
        archive.find_sparta_files([1], 58024.3, 58024.41, deadline=Deadline(0))


def test_download_sets_timeout_per_file(tmp_path, monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests.Session, "request", lambda self, *a, **k: calls.append(k.get("timeout")))

    def fake(self, datasets, destination, **kwargs):
        self._session.get("http://example.invalid/x")
        (tmp_path / f"{datasets[0]}.fits.Z").write_bytes(b"")

    monkeypatch.setattr("astroquery.eso.core.EsoClass.retrieve_data", fake)
    paths = download_sparta_files(["SPHER.A", "SPHER.B"], tmp_path, deadline=Deadline(30))
    assert len(paths) == 2
    assert len(calls) == 2 and all(0 < t <= 30 for t in calls)


def test_download_stops_when_deadline_expires(tmp_path, monkeypatch):
    deadline = Deadline(60)
    fetched = []

    def fake(self, datasets, destination, **kwargs):
        fetched.append(datasets[0])
        (tmp_path / f"{datasets[0]}.fits.Z").write_bytes(b"")
        deadline._end = 0

    monkeypatch.setattr("astroquery.eso.core.EsoClass.retrieve_data", fake)
    with pytest.raises(SourceFailed) as info:
        download_sparta_files(["SPHER.A", "SPHER.B"], tmp_path, deadline=deadline)
    assert fetched == ["SPHER.A"]
    assert info.value.present == [tmp_path / "SPHER.A.fits.Z"]


def test_partial_download_failure_keeps_present_paths(tmp_path, monkeypatch):
    def fake(self, datasets, destination, **kwargs):
        if datasets[0] == "SPHER.B":
            raise RuntimeError("connection reset")
        (tmp_path / f"{datasets[0]}.fits.Z").write_bytes(b"")

    monkeypatch.setattr("astroquery.eso.core.EsoClass.retrieve_data", fake)
    with pytest.raises(SourceFailed) as info:
        download_sparta_files(["SPHER.A", "SPHER.B"], tmp_path, deadline=Deadline(60))
    assert info.value.present == [tmp_path / "SPHER.A.fits.Z"]

import urllib.error
from pathlib import Path

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


def _http_error(status):
    import requests

    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


class _Archive:
    """Stand-in for astroquery's per-file download: serves ``public`` always and ``private`` after a login."""

    def __init__(self, monkeypatch, public=(), private=()):
        self.public, self.private = set(public), set(private)
        self.logins = []
        archive = self

        def download(eso, file_link, destination, overwrite):
            dp_id = file_link.rsplit("/", 1)[-1]
            if dp_id in archive.public or (dp_id in archive.private and archive.logins):
                path = Path(destination) / f"{dp_id}.fits.Z"
                path.write_bytes(b"")
                return str(path), True
            raise _http_error(401)

        def login(eso, username=None, store_password=False, **kwargs):
            archive.logins.append(username)
            return True

        monkeypatch.setattr("astroquery.eso.core.EsoClass._download_eso_file", download)
        monkeypatch.setattr("astroquery.eso.core.EsoClass.login", login)
        monkeypatch.setattr("getpass.getpass", lambda *a, **k: pytest.fail("password prompt"))


def test_file_that_astroquery_fails_to_download_raises(tmp_path, monkeypatch):
    _Archive(monkeypatch, public={"SPHER.A"})
    with pytest.raises(SourceFailed) as info:
        download_sparta_files(["SPHER.A", "SPHER.B"], tmp_path, deadline=Deadline(60))
    assert "not downloaded: SPHER.B;" in str(info.value)
    assert "eso_username" in str(info.value)
    assert info.value.present == [tmp_path / "SPHER.A.fits.Z"]


def test_login_is_skipped_without_a_stored_password(tmp_path, monkeypatch):
    archive = _Archive(monkeypatch, private={"SPHER.B"})
    monkeypatch.setattr("keyring.get_password", lambda service, user: None)
    with pytest.raises(SourceFailed) as info:
        download_sparta_files(["SPHER.B"], tmp_path, deadline=Deadline(60), eso_username="someone")
    assert archive.logins == []
    assert "no ESO password for someone" in str(info.value)


def test_login_with_a_stored_password_fetches_proprietary_files(tmp_path, monkeypatch):
    from spherical.conditions.archive import KEYRING_SERVICE

    archive = _Archive(monkeypatch, public={"SPHER.A"}, private={"SPHER.B"})
    seen = []
    monkeypatch.setattr("keyring.get_password", lambda service, user: seen.append(service) or "secret")
    paths = download_sparta_files(["SPHER.A", "SPHER.B"], tmp_path, deadline=Deadline(60), eso_username="someone")
    assert paths == [tmp_path / "SPHER.A.fits.Z", tmp_path / "SPHER.B.fits.Z"]
    assert archive.logins == ["someone"]
    assert seen == [KEYRING_SERVICE]


def test_query_without_ob_ids_skips_the_network(monkeypatch):
    from spherical.conditions import archive

    assert sparta_query([], 58024.3, 58024.4) is None
    monkeypatch.setattr(archive.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    assert archive.find_sparta_files([], 58024.3, 58024.4, deadline=Deadline(5)).empty


def test_unfiltered_query_keeps_rows_without_data_type(monkeypatch):
    from spherical.conditions import archive

    reply = 'dp_id,mjd_obs,ob_id,dp_type\nSPHER.A,58024.3,1,"OBJECT,AO"\nSPHER.B,58024.31,2,\nSPHER.C,58024.32,1,OBJECT\n'
    monkeypatch.setattr(archive.urllib.request, "urlopen", lambda url, timeout: _Reply(reply))
    loose = archive.find_sparta_files([1], 58024.3, 58024.4, deadline=Deadline(5), filtered=False)
    assert list(loose["dp_id"]) == ["SPHER.A", "SPHER.B"]
    strict = archive.find_sparta_files([1], 58024.3, 58024.4, deadline=Deadline(5))
    assert list(strict["dp_id"]) == ["SPHER.A"]


def test_ambient_endpoint_follows_the_site_monitor_upgrade(monkeypatch):
    urls = []

    def fake(url, timeout):
        urls.append(url)
        return _Reply("")

    monkeypatch.setattr(ambient.urllib.request, "urlopen", fake)
    before, after = Time("2015-09-25T08:00").mjd, Time("2017-09-28T08:00").mjd
    ambient.query_ambient("dimm", before, before + 0.1, deadline=Deadline(5))
    ambient.query_ambient("dimm", after, after + 0.1, deadline=Deadline(5))
    ambient.query_ambient("meteo", before, before + 0.1, deadline=Deadline(5))
    assert "/historical_ambient_paranal/" in urls[0] and "tab_rfl=1" in urls[0]
    assert "/dimm_paranal/" in urls[1]
    assert "/meteo_paranal/" in urls[2]
    assert "2016-04-05" in ambient.unavailable_reason("mass", before, before + 0.1)
    assert ambient.unavailable_reason("mass", after, after + 0.1) is None
    assert ambient.unavailable_reason("dimm", before, before + 0.1) is None
    with pytest.raises(SourceFailed):
        ambient.query_ambient("mass", before, before + 0.1, deadline=Deadline(5))

"""SIMBAD server choice and failover to the mirrors (#260). All offline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from astropy.table import Table

from spherical.database import simbad_servers, target_table

CDS = "simbad.cds.unistra.fr"
HARVARD = "simbad.harvard.edu"
AVAILABLE = '<availability xmlns="http://www.ivoa.net/xml/VOSIAvailability/v1.0">\n\t<available>true</available>\n'
# Captured from simbad.harvard.edu with simbad_tap_query.adql for one upload row.
TAP_RESULT_51ERI = Path(__file__).parent / "data" / "simbad_tap_51eri.ecsv"


def _response(ok=True, text=AVAILABLE):
    return MagicMock(ok=ok, text=text, status_code=200 if ok else 503)


def _upload_table():
    return Table({"OBJECT": ["51 Eri"], "RA": [69.4005], "DEC": [-2.4735], "MJD_OBS": [57000.0]})


@pytest.fixture(autouse=True)
def fresh_probe():
    simbad_servers.is_available.cache_clear()
    yield
    simbad_servers.is_available.cache_clear()


@pytest.fixture
def known_servers(monkeypatch):
    monkeypatch.setattr(simbad_servers.conf, "servers_list", [CDS, HARVARD])
    monkeypatch.setattr(simbad_servers.Simbad, "server", CDS)


class TestServerOrder:
    def test_shared_simbad_server_first(self, known_servers):
        assert simbad_servers.simbad_servers() == [CDS, HARVARD]

    def test_setting_shared_simbad_server_prefers_the_mirror(self, known_servers):
        simbad_servers.Simbad.server = HARVARD
        assert simbad_servers.simbad_servers() == [HARVARD, CDS]


class TestAvailability:
    def test_available(self):
        with patch.object(simbad_servers.requests, "get", return_value=_response()) as get:
            assert simbad_servers.is_available(HARVARD)
        get.assert_called_once_with(
            f"https://{HARVARD}/simbad/sim-tap/availability", timeout=simbad_servers.AVAILABILITY_TIMEOUT
        )

    def test_service_reports_unavailable(self):
        text = "<availability><available>false</available></availability>"
        with patch.object(simbad_servers.requests, "get", return_value=_response(text=text)):
            assert not simbad_servers.is_available(CDS)

    def test_unreachable(self):
        with patch.object(simbad_servers.requests, "get", side_effect=requests.ConnectTimeout("timed out")):
            assert not simbad_servers.is_available(CDS)

    def test_only_answering_servers_are_listed(self, known_servers):
        with patch.object(simbad_servers, "is_available", side_effect=lambda server: server == HARVARD):
            assert list(simbad_servers.available_simbad_servers()) == [HARVARD]

    def test_each_server_is_probed_once_per_session(self, known_servers):
        with patch.object(simbad_servers.requests, "get", side_effect=requests.ConnectTimeout("timed out")) as get:
            for _ in range(3):
                assert list(simbad_servers.available_simbad_servers()) == []
        assert get.call_count == 2

    def test_later_servers_are_probed_only_when_needed(self, known_servers):
        simbad_servers.Simbad.server = HARVARD
        with patch.object(simbad_servers.requests, "get", return_value=_response()) as get:
            assert next(simbad_servers.available_simbad_servers()) == HARVARD
        get.assert_called_once()

    def test_client_is_bound_to_server_without_touching_shared_simbad(self, known_servers):
        from astroquery.simbad import Simbad

        before = Simbad.server
        assert simbad_servers.simbad_client(HARVARD).server == HARVARD
        assert Simbad.server == before


class TestQueryObject:
    def test_falls_through_to_next_server(self, known_servers):
        clients = {CDS: MagicMock(), HARVARD: MagicMock()}
        clients[CDS].query_object.side_effect = ConnectionError("down")
        clients[HARVARD].query_object.return_value = Table({"main_id": ["*  51 Eri"]})
        with patch.object(simbad_servers, "is_available", return_value=True), \
             patch.object(simbad_servers, "simbad_client", side_effect=clients.get):
            result = simbad_servers.query_object("51 Eri")
        assert result["main_id"][0] == "*  51 Eri"

    def test_none_with_warning_when_no_server_answers(self, known_servers):
        with patch.object(simbad_servers, "is_available", return_value=False), \
             pytest.warns(UserWarning, match="No SIMBAD server"):
            assert simbad_servers.query_object("51 Eri") is None


class TestBatchFailover:
    def _clients(self, failing):
        clients = {}
        for server in (CDS, HARVARD):
            client = MagicMock(server=server)
            if server in failing:
                client.query_tap.side_effect = ConnectionError("down")
            else:
                client.query_tap.return_value = Table.read(TAP_RESULT_51ERI)
            clients[server] = client
        return clients

    def _query(self, clients, available=(CDS, HARVARD)):
        with patch.object(simbad_servers, "available_simbad_servers", return_value=iter(available)), \
             patch.object(simbad_servers, "simbad_client", side_effect=clients.get), \
             patch.object(target_table.time, "sleep"):
            return target_table.query_SIMBAD_for_names(_upload_table(), min_delay=0.0)

    def test_failed_batch_is_retried_on_the_mirror(self):
        clients = self._clients(failing={CDS})
        result, _ = self._query(clients)
        assert clients[HARVARD].query_tap.call_count == 1
        assert result["MAIN_ID"][0] == "*  51 Eri"
        assert result.meta["SIMBAD_SERVERS"] == [HARVARD]

    def test_unreachable_server_is_skipped(self):
        clients = self._clients(failing=set())
        result, _ = self._query(clients, available=[HARVARD])
        clients[CDS].query_tap.assert_not_called()
        assert result.meta["SIMBAD_SERVERS"] == [HARVARD]

    def test_batch_failing_everywhere_raises(self):
        with pytest.raises(ConnectionError, match="every available server"):
            self._query(self._clients(failing={CDS, HARVARD}))

    def test_no_available_server_raises(self):
        with patch.object(simbad_servers, "available_simbad_servers", return_value=iter([])), \
             pytest.raises(ConnectionError, match="No SIMBAD server is available"):
            target_table.query_SIMBAD_for_names(_upload_table())

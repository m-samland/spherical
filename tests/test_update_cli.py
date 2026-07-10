from unittest.mock import patch

from spherical.database import provenance as prov
from spherical.scripts import update_database as cli


def _fake_update_writing(enrichment):
    """Return an update_database stand-in that writes provenance for the mode."""
    def fake_update(dest, instrument, **kw):
        rec = prov.TableProvenance(instrument=instrument, mode=instrument, enrichment=enrichment)
        prov.write_provenance(dest, {instrument: rec})
        return {instrument: rec}
    return fake_update


_HEALTHY = {
    "gaia": {"status": "ok", "n_valid_ids": 100, "n_matched": 70, "frac": 0.70},
    "moca": {"status": "ok", "n_valid_ids": 100, "n_matched": 80, "frac": 0.80, "tier2_ok": True},
}
_BREACHING = {
    "gaia": {"status": "ok", "n_valid_ids": 100, "n_matched": 10, "frac": 0.10},
    "moca": {"status": "ok", "n_valid_ids": 100, "n_matched": 80, "frac": 0.80, "tier2_ok": True},
}


def test_healthy_enrichment_returns_zero(tmp_path):
    with patch.object(cli.build, "update_database", side_effect=_fake_update_writing(_HEALTHY)):
        rc = cli.main(["--dest", str(tmp_path), "--instrument", "ifs", "--start-date", "2016-09-15"])
    assert rc == 0


def test_enrichment_below_floor_returns_nonzero(tmp_path):
    with patch.object(cli.build, "update_database", side_effect=_fake_update_writing(_BREACHING)):
        rc = cli.main(["--dest", str(tmp_path), "--instrument", "ifs", "--start-date", "2016-09-15"])
    assert rc == 1


def test_parser_defaults():
    args = cli.build_parser().parse_args(["--dest", "/tmp/db"])
    assert args.instrument == "all"
    assert args.overlap_days == 7
    assert args.skip_sam is False
    assert args.enrich_only is False


def test_all_dispatches_both_instruments(tmp_path):
    with patch.object(cli.build, "update_database") as upd:
        rc = cli.main(["--dest", str(tmp_path), "--instrument", "all",
                       "--start-date", "2016-09-15", "--end-date", "2016-09-16"])
    assert rc == 0
    called = {c.args[1] for c in upd.call_args_list}
    assert called == {"ifs", "irdis"}


def test_enrich_only_calls_enrich_for_each_mode(tmp_path):
    with patch.object(cli.build, "enrich_tables") as enr:
        rc = cli.main(["--dest", str(tmp_path), "--instrument", "ifs", "--enrich-only"])
    assert rc == 0
    # ifs standard + ifs_sam
    assert enr.call_count == 2


def test_no_enrich_flag_forwarded(tmp_path):
    with patch.object(cli.build, "update_database") as upd:
        cli.main(["--dest", str(tmp_path), "--instrument", "ifs",
                  "--start-date", "2016-09-15", "--no-enrich"])
    assert upd.call_args.kwargs["enrich"] is False


def test_mode_enriches_single_mode(tmp_path):
    with patch.object(cli.build, "enrich_tables") as enr:
        rc = cli.main(["--dest", str(tmp_path), "--mode", "irdis", "--enrich-only"])
    assert rc == 0
    assert enr.call_count == 1
    call = enr.call_args
    assert call.args[1] == "irdis"
    assert call.kwargs["polarimetry"] is False
    assert call.kwargs["sparse_aperture_masking"] is False


def test_mode_polarimetry_maps_correctly(tmp_path):
    with patch.object(cli.build, "enrich_tables") as enr:
        cli.main(["--dest", str(tmp_path), "--mode", "irdis_polarimetry", "--enrich-only"])
    assert enr.call_args.kwargs["polarimetry"] is True
    assert enr.call_args.kwargs["sparse_aperture_masking"] is False


def test_mode_sam_maps_correctly(tmp_path):
    with patch.object(cli.build, "enrich_tables") as enr:
        cli.main(["--dest", str(tmp_path), "--mode", "ifs_sam", "--enrich-only"])
    assert enr.call_args.args[1] == "ifs"
    assert enr.call_args.kwargs["sparse_aperture_masking"] is True


def test_mode_without_enrich_only_errors(tmp_path):
    with patch.object(cli.build, "enrich_tables") as enr, \
         patch.object(cli.build, "update_database") as upd:
        rc = cli.main(["--dest", str(tmp_path), "--mode", "irdis"])
    assert rc == 1
    assert enr.call_count == 0
    assert upd.call_count == 0

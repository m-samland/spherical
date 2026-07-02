from unittest.mock import patch

from spherical.scripts import update_database as cli


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

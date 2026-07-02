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

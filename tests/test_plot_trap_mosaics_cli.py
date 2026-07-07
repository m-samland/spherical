import argparse
import sys
from pathlib import Path

import pytest

from spherical.scripts import plot_trap_mosaics as cli


def test_parser_defaults():
    args = cli.build_parser().parse_args(["/some/base"])
    assert args.base_path == Path("/some/base")
    assert args.content == "combined"
    assert args.templates == ["flat", "L-type", "T-type"]
    assert args.format == "pdf"
    assert args.batch_size is None
    assert args.auto_scale is False
    assert args.dpi == 300
    assert args.output is None
    assert args.database_dir is None
    assert args.suffix is None
    assert args.snr_min is None
    assert args.snr_max is None


def test_parser_overrides():
    args = cli.build_parser().parse_args(
        [
            "/base",
            "--content", "spectrum",
            "--templates", "flat", "T-type",
            "--format", "png",
            "--batch-size", "25",
            "--auto-scale",
            "--dpi", "150",
            "--output", "/out",
            "--database-dir", "/db",
            "--suffix", "young",
            "--snr-min", "5",
            "--snr-max", "20",
        ]
    )
    assert args.content == "spectrum"
    assert args.templates == ["flat", "T-type"]
    assert args.format == "png"
    assert args.batch_size == 25
    assert args.auto_scale is True
    assert args.dpi == 150
    assert args.output == Path("/out")
    assert args.database_dir == Path("/db")
    assert args.suffix == "young"
    assert args.snr_min == 5.0
    assert args.snr_max == 20.0


def test_resolve_output_dir_default():
    assert cli.resolve_output_dir(Path("/base"), None) == Path("/base/mosaics")


def test_resolve_output_dir_override():
    assert cli.resolve_output_dir(Path("/base"), Path("/custom")) == Path("/custom")


def test_build_output_path_basic():
    assert cli.build_output_path(Path("/out"), "combined", "flat", None, "pdf") == Path(
        "/out/combined_mosaic_flat.pdf"
    )


def test_build_output_path_suffix():
    assert cli.build_output_path(Path("/out"), "detection", "L-type", "young", "png") == Path(
        "/out/detection_mosaic_L-type_young.png"
    )


def test_select_plot_function_single():
    assert cli.select_plot_function("combined", batched=False) is cli.mosaic.plot_combined_mosaic


def test_select_plot_function_batched():
    assert cli.select_plot_function("spectrum", batched=True) is cli.mosaic.plot_spectrum_mosaic_batched


def _make_args(**overrides):
    defaults = dict(
        base_path=Path("/base"),
        content="combined",
        templates=["flat"],
        format="pdf",
        batch_size=None,
        auto_scale=False,
        dpi=300,
        output=None,
        database_dir=None,
        suffix=None,
        snr_min=None,
        snr_max=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_obs_table_no_dir():
    assert cli.load_obs_table(None) is None


def test_load_obs_table_missing_file(tmp_path):
    assert cli.load_obs_table(tmp_path) is None


def test_load_obs_table_found(tmp_path, monkeypatch):
    (tmp_path / cli.OBS_TABLE_FILENAME).write_text("x")
    monkeypatch.setattr(cli.mosaic, "load_observation_table", lambda p: "TABLE")
    assert cli.load_obs_table(tmp_path) == "TABLE"


def test_run_missing_base(tmp_path):
    assert cli.run(_make_args(base_path=tmp_path / "nope")) == 1


def test_run_no_combinations(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.mosaic, "get_all_combinations", lambda bp: [])
    assert cli.run(_make_args(base_path=tmp_path)) == 1


def test_run_happy_path(tmp_path, monkeypatch):
    calls = []

    def fake_combined(base_path, template_type, output_path, **kwargs):
        calls.append((template_type, output_path, kwargs))
        return "FIG"

    monkeypatch.setattr(cli.mosaic, "get_all_combinations", lambda bp: [("t", "m", "d")])
    monkeypatch.setattr(cli.mosaic, "plot_combined_mosaic", fake_combined)
    monkeypatch.setattr(cli.plt, "close", lambda fig: None)

    rc = cli.run(_make_args(base_path=tmp_path, templates=["flat", "L-type"]))

    assert rc == 0
    assert (tmp_path / "mosaics").is_dir()
    assert [c[0] for c in calls] == ["flat", "L-type"]
    tt, out, kwargs = calls[0]
    assert out == tmp_path / "mosaics" / "combined_mosaic_flat.pdf"
    assert kwargs["observation_table"] is None
    assert kwargs["dpi"] == 300
    assert kwargs["auto_scale"] is False
    assert kwargs["snr_min"] is None
    assert kwargs["snr_max"] is None


def test_run_spectrum_omits_auto_scale(tmp_path, monkeypatch):
    captured = {}

    def fake_spectrum(base_path, template_type, output_path, **kwargs):
        captured.update(kwargs)
        return "FIG"

    monkeypatch.setattr(cli.mosaic, "get_all_combinations", lambda bp: [("t", "m", "d")])
    monkeypatch.setattr(cli.mosaic, "plot_spectrum_mosaic", fake_spectrum)
    monkeypatch.setattr(cli.plt, "close", lambda fig: None)

    rc = cli.run(_make_args(base_path=tmp_path, content="spectrum", auto_scale=True))

    assert rc == 0
    assert "auto_scale" not in captured


def test_run_batched_dispatch(tmp_path, monkeypatch):
    calls = []

    def fake_batched(base_path, template_type, batch_size, output_dir, **kwargs):
        calls.append((template_type, batch_size, output_dir, kwargs))
        return ["F1", "F2"]

    monkeypatch.setattr(cli.mosaic, "get_all_combinations", lambda bp: [("t", "m", "d")])
    monkeypatch.setattr(cli.mosaic, "plot_combined_mosaic_batched", fake_batched)
    monkeypatch.setattr(cli.plt, "close", lambda fig: None)

    rc = cli.run(
        _make_args(base_path=tmp_path, batch_size=30, format="pdf", suffix="young")
    )

    assert rc == 0
    tt, bs, od, kwargs = calls[0]
    assert (tt, bs, od) == ("flat", 30, tmp_path / "mosaics")
    assert kwargs["output_format"] == "pdf"
    assert kwargs["suffix"] == "young"


def test_main_exits_with_run_return_code(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.mosaic, "get_all_combinations", lambda bp: [])
    monkeypatch.setattr(sys, "argv", ["plot_trap_mosaics", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1

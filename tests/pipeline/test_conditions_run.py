"""Orchestration, caching and coverage summary of the observing-conditions step."""

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.time import Time
from test_conditions_sparta import T0, _dtts_star, _dtts_table, _table, _write

from spherical.conditions import run as run_module
from spherical.conditions.archive import SourceFailed
from spherical.conditions.run import (
    COVERAGE_FILE,
    ConditionsSettings,
    run_conditions,
)

LOGGER = logging.getLogger("test_conditions_run")
AMBIENT_CSV = "Date time,Value\n2017-09-28T08:00:00,1.0\n2017-09-28T08:05:00,2.0\n"


def _mjd(unix):
    return Time(np.asarray(unix, dtype=float), format="unix").mjd


def _sparta_hdus(first, last):
    """Tables with every sample stamp from ``first`` to ``last`` (unix seconds).

    The standard deviations change from row to row, so no row repeats the previous one and none is dropped.
    """
    vis = np.arange(first, last + 1, 10.0)
    atm = np.arange(first, last + 1, 20.0)
    ir = np.arange(first, last + 1, 30.0)
    dtts = np.arange(first, last + 1, 2.0)
    return [
        _table(
            "AtmPerfParams",
            atm,
            R0=np.full(len(atm), 0.1),
            T0=np.arange(len(atm), dtype=float),
            L0=0 * atm + 25,
            WindSpeed=np.full(len(atm), 10.0),
            StrehlRatio=np.full(len(atm), 0.8),
        ),
        _table("VisLoopParams", vis, Flux_avg=np.full(len(vis), 17000.0), Flux_std=np.arange(len(vis), dtype=float)),
        _table("IRLoopParams", ir, Flux_avg=np.full(len(ir), 2.0), Flux_std=np.arange(len(ir), dtype=float)),
        _dtts_table(_dtts_star(len(dtts)), dtts),
    ]


def _frames(first, count=10, dit=32.0, ob_id=1837295, waffle=None):
    start = first + dit * np.arange(count)
    table = pd.DataFrame({"MJD START": _mjd(start), "MJD END": _mjd(start + dit), "OBS ID": ob_id})
    if waffle is not None:
        table["OCS WAFFLE AMPL"] = waffle
    return table


class Network:
    """Stand-ins for the archive and ambient functions that count their calls."""

    def __init__(self, tmp_path, monkeypatch, files=(), ambient_fail=()):
        self.remote = tmp_path / "remote"
        self.remote.mkdir()
        self.files = list(files)
        self.ambient_fail = set(ambient_fail)
        self.calls = {"find": [], "download": 0, "ambient": []}
        monkeypatch.setattr(run_module, "find_sparta_files", self.find)
        monkeypatch.setattr(run_module, "download_sparta_files", self.download)
        monkeypatch.setattr(run_module, "query_ambient", self.ambient)

    def add(self, name, hdus, **kwargs):
        path = _write(self.remote / f"{name}.fits", hdus, **kwargs)
        self.files.append((name, float(_mjd(T0))))
        return path

    def find(self, ob_ids, mjd_start, mjd_end, *, deadline, filtered=True):
        self.calls["find"].append({"ob_ids": list(ob_ids), "window": (mjd_start, mjd_end), "filtered": filtered})
        if not filtered:
            return pd.DataFrame({"dp_id": [], "mjd_obs": [], "ob_id": [], "dp_type": []})
        names = [n for n, _ in self.files]
        return pd.DataFrame(
            {"dp_id": names, "mjd_obs": [m for _, m in self.files], "ob_id": ob_ids[0], "dp_type": "OBJECT,AO"}
        )

    def download(self, dp_ids, destination, *, deadline, eso_username=None, store_password=False):
        self.calls["download"] += 1
        for dp_id in dp_ids:
            if not (destination / f"{dp_id}.fits").exists():
                shutil.copy(self.remote / f"{dp_id}.fits", destination / f"{dp_id}.fits")
        return [destination / f"{d}.fits" for d in dp_ids]

    def ambient(self, name, mjd_start, mjd_end, *, deadline):
        self.calls["ambient"].append(name)
        if name in self.ambient_fail:
            raise SourceFailed(f"{name} down")
        return AMBIENT_CSV


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "sparta", tmp_path / "out"


def _run(frames, dirs, **settings):
    sparta_dir, out = dirs
    return run_conditions(frames, sparta_dir, out, ConditionsSettings(**settings), LOGGER)


def test_products_and_complete_summary(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    frames = {"CENTER": _frames(T0 + 20), "FLUX": _frames(T0 + 5, count=2, dit=2.0)}
    frames["CENTER"]["OCS WAFFLE AMPL"] = 0.2
    summary = _run(frames, dirs)
    out = dirs[1]
    assert summary.complete
    assert (out / COVERAGE_FILE).exists()
    assert summary.sources == {"sparta": "found", "mass": "found", "dimm": "found", "meteo": "found", "lhatpro": "found"}
    center = pd.read_csv(out / "matched_center.csv")
    assert len(center) == len(frames["CENTER"])
    assert (center["flag_visloop"] == "direct").all()
    assert center["visloop_flux_subap"].iloc[0] == pytest.approx(1.0)
    assert center["atm_seeing"].notna().all()
    assert len(pd.read_csv(out / "matched_flux.csv")) == 2
    config = pd.read_csv(out / "sparta_config.csv")
    assert len(config) == 1
    assert config["waffle_amplitude"].iloc[0] == pytest.approx(0.2)
    assert config["wfs_gain"].iloc[0] == 1000
    for name in ("atm", "visloop", "irloop", "dtts"):
        assert (out / f"sparta_{name}.csv").exists()
    assert summary.direct_fraction["visloop_flux_subap"] == 1.0
    assert not list(out.glob("*.fits"))


def test_failed_source_writes_non_gating_summary(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch, ambient_fail={"meteo"})
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    out = dirs[1]
    out.mkdir()
    (out / COVERAGE_FILE).write_text("{}")
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert not summary.complete
    assert summary.sources["meteo"] == "failed"
    assert (out / "coverage_incomplete.json").exists()
    assert not (out / COVERAGE_FILE).exists()
    assert (out / "matched_center.csv").exists()
    assert json.loads((out / "coverage_incomplete.json").read_text())["messages"]["meteo"] == "meteo down"


def test_empty_answer_is_not_failure(tmp_path, monkeypatch, dirs):
    Network(tmp_path, monkeypatch)
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert summary.sources["sparta"] == "empty"
    assert summary.complete
    assert (dirs[1] / COVERAGE_FILE).exists()
    center = pd.read_csv(dirs[1] / "matched_center.csv")
    assert (center["flag_visloop"] == "gap").all()
    assert (center["flag_dtts"] == "gap").all()


def test_time_budget_marks_remaining_sources_failed(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs, time_budget_s=0)
    assert set(summary.sources.values()) == {"failed"}
    assert set(summary.sources) == {"sparta", "mass", "dimm", "meteo", "lhatpro"}
    assert summary.messages["mass"] == "time budget exceeded"
    assert (dirs[1] / "coverage_incomplete.json").exists()
    assert net.calls["ambient"] == []


def test_cache_is_reused(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    frames = {"CENTER": _frames(T0 + 20)}
    _run(frames, dirs)
    first = {"find": len(net.calls["find"]), "download": net.calls["download"], "ambient": len(net.calls["ambient"])}
    assert first["download"] == 1 and first["ambient"] == 4
    assert (dirs[0] / "sparta_files.csv").exists() and (dirs[0] / "ambient_mass.csv").exists()
    summary = _run(frames, dirs)
    assert summary.complete
    assert (len(net.calls["find"]), net.calls["download"], len(net.calls["ambient"])) == (
        first["find"],
        1,
        4,
    )
    _run(frames, dirs, refresh_cache=True)
    assert len(net.calls["find"]) > first["find"]
    assert len(net.calls["ambient"]) == 8


def test_configuration_change_starts_new_segment(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 300))
    net.add("SPHER.B", _sparta_hdus(T0 + 330, T0 + 600))
    fits.setval(net.remote / "SPHER.B.fits", "HIERARCH ESO AOS VISWFS MODE", value="GAIN_100_FREQ_1380Hz")
    frames = {"CENTER": _frames(T0 + 250, count=3, dit=32.0)}
    frames["CENTER"].loc[1, ["MJD START", "MJD END"]] = _mjd([T0 + 311, T0 + 318])
    _run(frames, dirs)
    config = pd.read_csv(dirs[1] / "sparta_config.csv")
    assert sorted(config["segment"]) == [0, 1]
    center = pd.read_csv(dirs[1] / "matched_center.csv")
    assert list(center["segment"][[0, 2]]) == [0, 1]
    assert center["flag_visloop"][1] == "gap"
    assert np.isnan(center["visloop_flux_subap"][1])


def test_gap_triggers_unfiltered_query(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 200))
    net.add("SPHER.B", _sparta_hdus(T0 + 1100, T0 + 1300))
    frames = {"CENTER": _frames(T0 + 20, count=36)}
    summary = _run(frames, dirs)
    unfiltered = [c for c in net.calls["find"] if not c["filtered"]]
    assert len(unfiltered) == 1
    assert len(summary.gaps) == 1
    gap = summary.gaps[0]
    assert (gap["mjd_end"] - gap["mjd_start"]) * 86400 == pytest.approx(900, abs=15)
    assert gap["unfiltered_files"] == []
    assert unfiltered[0]["window"] == pytest.approx((gap["mjd_start"], gap["mjd_end"]))
    assert json.loads((dirs[1] / COVERAGE_FILE).read_text())["gaps"][0]["unfiltered_files"] == []


def test_corrupt_sparta_file_is_removed_and_fetched_again(tmp_path, monkeypatch, dirs, caplog):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    sparta_dir = dirs[0]
    sparta_dir.mkdir()
    good = (net.remote / "SPHER.A.fits").read_bytes()
    (sparta_dir / "SPHER.A.fits").write_bytes(good[: len(good) // 2])
    (net.remote / "SPHER.BAD.fits").write_bytes(b"")
    net.files.append(("SPHER.BAD", float(_mjd(T0))))
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert not summary.complete
    assert summary.sources["sparta"] == "failed"
    assert "SPHER.A" in summary.messages["sparta"] and "SPHER.BAD" in caplog.text
    assert (dirs[1] / "coverage_incomplete.json").exists() and not (dirs[1] / COVERAGE_FILE).exists()
    assert not (sparta_dir / "SPHER.A.fits").exists()
    downloads = []
    original = net.download

    def counting(dp_ids, destination, **kwargs):
        downloads.extend(dp_ids)
        return original(dp_ids, destination, **kwargs)

    monkeypatch.setattr(run_module, "download_sparta_files", counting)
    (net.remote / "SPHER.BAD.fits").unlink()
    net.files = [f for f in net.files if f[0] != "SPHER.BAD"]
    (dirs[0] / "sparta_files.csv").unlink()
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert downloads == ["SPHER.A"]
    assert summary.complete and summary.sparta_files == ["SPHER.A"]


def test_direct_fraction_counts_non_nan_science_frames_only(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    hdus = _sparta_hdus(T0, T0 + 600)
    net.add("SPHER.A", hdus)
    path = net.remote / "SPHER.A.fits"
    with fits.open(path, mode="update") as hdul:
        hdul["IRLoopParams"].data["Flux_avg"][:] = np.nan
        hdul["IRLoopParams"].data["Flux_avg"][:3] = 2.0
    frames = {"CENTER": _frames(T0 + 20, count=4, dit=32.0), "FLUX": _frames(T0 + 5, count=2, dit=2.0)}
    summary = _run(frames, dirs)
    assert summary.direct_fraction["visloop_flux_subap"] == 1.0
    center = pd.read_csv(dirs[1] / "matched_center.csv")
    expected = center["irloop_Flux_avg"].notna().sum() / 4
    assert expected == 0.5
    assert summary.direct_fraction["irloop_Flux_avg"] == pytest.approx(expected)


def test_unfiltered_gap_query_window_is_extended_by_margin(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 200))
    net.add("SPHER.B", _sparta_hdus(T0 + 1100, T0 + 1300))
    summary = _run({"CENTER": _frames(T0 + 20, count=36)}, dirs, margin_s=300)
    call = [c for c in net.calls["find"] if not c["filtered"]][0]
    assert call["window"][1] == pytest.approx(summary.gaps[0]["mjd_end"] + 300 / 86400)
    assert call["window"][0] == pytest.approx(summary.gaps[0]["mjd_start"])


def test_all_obs_ids_and_margin_reach_the_query(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    frames = {"CENTER": _frames(T0 + 20, count=2), "FLUX": _frames(T0, count=1, dit=2.0, ob_id=1837296)}
    _run(frames, dirs, margin_s=600)
    call = net.calls["find"][0]
    assert call["ob_ids"] == [1837295, 1837296]
    assert call["window"][0] == pytest.approx(float(_mjd(T0)) - 600 / 86400)


def test_missing_config_file_reports_nan_waffle(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert np.isnan(pd.read_csv(dirs[1] / "sparta_config.csv")["waffle_amplitude"].iloc[0])


def test_lzw_compressed_sparta_file_is_read(tmp_path, monkeypatch, dirs):
    monkeypatch.setenv("PATH", "")
    Network(tmp_path, monkeypatch, files=[("SPHER.A", float(_mjd(T0)))])
    sparta_dir = dirs[0]
    sparta_dir.mkdir()
    # Synthetic SPARTA tables compressed once with ncompress, not at test time.
    fixture = Path(__file__).with_name("data") / "sparta-lzw.fits.Z"
    shutil.copyfile(fixture, sparta_dir / "SPHER.A.fits.Z")
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert summary.sparta_files == ["SPHER.A"]
    assert (sparta_dir / "SPHER.A.fits").exists()


def test_files_astroquery_could_not_download_fail_the_source(tmp_path, monkeypatch, dirs):
    from test_conditions_archive import _Archive

    from spherical.conditions import archive

    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    monkeypatch.setattr(run_module, "download_sparta_files", archive.download_sparta_files)
    _Archive(monkeypatch)
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert summary.sources["sparta"] == "failed"
    assert "SPHER.A" in summary.messages["sparta"]
    assert summary.missing_files == ["SPHER.A"]
    written = json.loads((dirs[1] / "coverage_incomplete.json").read_text())
    assert written["missing_files"] == ["SPHER.A"]
    assert not (dirs[1] / COVERAGE_FILE).exists()


def test_complete_but_unparsable_file_is_kept_and_does_not_fail(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    net.add("SPHER.EMPTY", [])
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert summary.complete and summary.sources["sparta"] == "found"
    assert summary.sparta_files == ["SPHER.A"]
    assert "no SPARTA table" in summary.unreadable_files["SPHER.EMPTY"]
    assert (dirs[0] / "SPHER.EMPTY.fits").exists()
    _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert net.calls["download"] == 1


def test_missing_tables_are_recorded_per_file(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600)[:3])
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert "IRPixelAvgFrame" in summary.missing_tables["SPHER.A"]
    assert "IRPixelAvgFrame" in json.loads((dirs[1] / COVERAGE_FILE).read_text())["missing_tables"]["SPHER.A"]


def test_missing_uncompresspy_keeps_the_compressed_file(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.files.append(("SPHER.Z", float(_mjd(T0))))
    sparta_dir = dirs[0]
    sparta_dir.mkdir()
    (sparta_dir / "SPHER.Z.fits.Z").write_bytes(b"\x1f\x9d" + bytes(2000))
    monkeypatch.setitem(sys.modules, "uncompresspy", None)
    summary = _run({"CENTER": _frames(T0 + 20)}, dirs)
    assert summary.sources["sparta"] == "failed"
    assert "uncompresspy" in summary.messages["sparta"]
    assert (sparta_dir / "SPHER.Z.fits.Z").exists()


def test_gaps_are_limited_to_frames(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 300))
    net.add("SPHER.B", _sparta_hdus(T0 + 1500, T0 + 1800))
    first, second = _frames(T0 + 20, count=8), _frames(T0 + 1520, count=8)
    summary = _run({"CENTER": pd.concat([first, second], ignore_index=True)}, dirs)
    assert summary.gaps == []
    assert not [c for c in net.calls["find"] if not c["filtered"]]


def test_unfiltered_hits_carry_ob_id_and_time(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 200))
    net.add("SPHER.B", _sparta_hdus(T0 + 1100, T0 + 1300))
    listed = net.find

    def find(ob_ids, mjd_start, mjd_end, *, deadline, filtered=True):
        if filtered:
            return listed(ob_ids, mjd_start, mjd_end, deadline=deadline, filtered=filtered)
        net.calls["find"].append({"filtered": False})
        return pd.DataFrame(
            {"dp_id": ["SPHER.A", "SPHER.X"], "mjd_obs": [58000.1, 58000.2], "ob_id": [1837295, 99], "dp_type": ""}
        )

    monkeypatch.setattr(run_module, "find_sparta_files", find)
    summary = _run({"CENTER": _frames(T0 + 20, count=36)}, dirs)
    assert summary.gaps[0]["unfiltered_files"] == [{"dp_id": "SPHER.X", "ob_id": 99, "mjd_obs": 58000.2}]


def test_gap_without_query_is_null(tmp_path, monkeypatch, dirs):
    Network(tmp_path, monkeypatch)
    summary = _run({"CENTER": _frames(T0 + 20, count=36)}, dirs, time_budget_s=0)
    assert summary.gaps and summary.gaps[0]["unfiltered_files"] is None
    assert json.loads((dirs[1] / "coverage_incomplete.json").read_text())["gaps"][0]["unfiltered_files"] is None


def test_matched_tables_carry_the_frame_times(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    net.add("SPHER.A", _sparta_hdus(T0, T0 + 600))
    frames = _frames(T0 + 20)
    _run({"CENTER": frames}, dirs)
    center = pd.read_csv(dirs[1] / "matched_center.csv")
    np.testing.assert_allclose(center["MJD START"], frames["MJD START"])
    np.testing.assert_allclose(center["MJD END"], frames["MJD END"])


def test_empty_frame_tables_give_an_empty_summary(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    summary = _run({"CENTER": _frames(T0 + 20).iloc[:0]}, dirs)
    assert summary.complete
    assert set(summary.sources.values()) == {"empty"}
    assert net.calls["find"] == [] and net.calls["ambient"] == []


def test_frames_without_ob_id_skip_the_sparta_listing(tmp_path, monkeypatch, dirs):
    net = Network(tmp_path, monkeypatch)
    summary = _run({"CENTER": _frames(T0 + 20).drop(columns="OBS ID")}, dirs)
    assert summary.sources["sparta"] == "empty"
    assert "OB ID" in summary.messages["sparta"]
    assert not [c for c in net.calls["find"] if c["filtered"]]

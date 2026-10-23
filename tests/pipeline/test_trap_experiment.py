"""Fake planets land where TRAP's model expects them and never touch the source run."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trap")
from astropy.io import fits  # noqa: E402
from trap.makesource import yx_position_in_cube  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "tutorials" / "runs" / "investigation"))
import trap_experiment as te  # noqa: E402


def test_relative_yx_north_up_east_left():
    assert te.relative_yx(10, 0) == pytest.approx((10, 0))
    assert te.relative_yx(10, 90) == pytest.approx((0, -10))
    y, x = te.relative_yx(37.0, 166.0)
    assert math.degrees(math.atan2(-x, y)) % 360 == pytest.approx(166.0)  # TRAP's position_angle


def test_parse_fake():
    assert te.parse_fake("37:238.8:2") == (37.0, 238.8, 2.0)


def _gaussian(size=21, sigma=1.5):
    yy, xx = np.indices((size, size)) - size // 2
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def test_inject_fakes_follows_trap_geometry():
    n_wave, n_frames, size = 2, 5, 101
    cube = np.zeros((n_wave, n_frames, size, size), dtype=np.float32)
    psf = np.stack([_gaussian(), 2 * _gaussian()])
    pa = np.linspace(-20, 20, n_frames)
    centers = np.zeros((n_wave, n_frames, 2)) + [50.3, 49.6]
    centers[:, :, 0] += np.linspace(0, 1, n_frames)  # a drifting star
    yx = te.relative_yx(30.0, 120.0)
    amplitude = np.ones((n_wave, n_frames))
    amplitude[:, 2] = 0.5
    out = te.inject_fakes(cube, psf, pa, centers, [(yx, [1e-3, 2e-3])], yx_anamorphism=[1.0062, 1.0],
                          right_handed=False, amplitude=amplitude)
    assert not cube.any()  # the input is untouched
    for w, contrast in enumerate([1e-3, 2e-3]):
        expected = yx_position_in_cube((size, size), yx, pa, centers[w], [1.0062, 1.0], False)
        for f in range(n_frames):
            peak = np.unravel_index(np.argmax(out[w, f]), out[w, f].shape)
            assert np.allclose(peak, np.round(expected[f]), atol=1)
            assert out[w, f].sum() == pytest.approx(contrast * amplitude[w, f] * psf[w].sum(), rel=1e-3)


def test_replace_with_copy_leaves_the_hard_link_twin(tmp_path):
    src = tmp_path / "src.fits"
    fits.writeto(src, np.zeros((3, 3), dtype=np.float32))
    twin = tmp_path / "twin.fits"
    os.link(src, twin)
    te.replace_with_copy(twin, np.ones((3, 3), dtype=np.float32), fits.getheader(src))
    assert os.stat(src).st_ino != os.stat(twin).st_ino
    assert not fits.getdata(src).any() and fits.getdata(twin).all()


def _trap_folder(exp_dir):
    folder = exp_dir / "reduction" / "IRDIS" / "trap" / "*_51_Eri" / "DB_K12" / "2015-09-24"
    folder.mkdir(parents=True)
    return folder


def test_detection_finished_needs_the_marker(tmp_path):
    assert not te.detection_finished(tmp_path)
    folder = _trap_folder(tmp_path)
    assert not te.detection_finished(tmp_path)
    (folder / ".run_trap_detection.done").touch()
    assert te.detection_finished(tmp_path)


def test_set_aside_crash_reports(tmp_path):
    folder = _trap_folder(tmp_path)
    (folder / "trap_crash_report.txt").write_text("boom")
    moved = te.set_aside_crash_reports(tmp_path)
    assert not (folder / "trap_crash_report.txt").exists() and len(moved) == 1
    assert (tmp_path / moved[0]).read_text() == "boom"
    (folder / "trap_crash_report.txt").write_text("again")  # a second resume keeps both
    second = te.set_aside_crash_reports(tmp_path)
    assert second != moved and (tmp_path / moved[0]).read_text() == "boom"
    assert (tmp_path / second[0]).read_text() == "again"


def test_restore_science_cubes_links_only_missing_cubes(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "coro_cube.fits").write_bytes(b"cube")
    (src / "center_cube.fits").write_bytes(b"center")
    (dst / "center_cube.fits").write_bytes(b"injected")
    te.restore_science_cubes(dst, src)
    assert os.stat(dst / "coro_cube.fits").st_ino == os.stat(src / "coro_cube.fits").st_ino
    assert (dst / "center_cube.fits").read_bytes() == b"injected"

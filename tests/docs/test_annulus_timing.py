"""The annulus timing run reuses a tutorial run's products without being able to change them."""

import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "tutorials" / "runs"))
import annulus_timing as at  # noqa: E402


def _tree(root):
    (root / "converted").mkdir(parents=True)
    (root / "converted" / "coro_cube.fits").write_bytes(b"x" * 8)
    (root / "reduction.jsonlog").write_text("{}\n")
    return root


def test_hardlink_tree_shares_files(tmp_path):
    src = _tree(tmp_path / "src")
    at.hardlink_tree(src, tmp_path / "dst")
    a, b = src / "converted" / "coro_cube.fits", tmp_path / "dst" / "converted" / "coro_cube.fits"
    assert os.stat(a).st_ino == os.stat(b).st_ino


def test_snapshot_notices_a_changed_file(tmp_path):
    src = _tree(tmp_path / "src")
    before = at.snapshot(src)
    (src / "converted" / "coro_cube.fits").write_bytes(b"y" * 9)
    assert at.changed_files(before, at.snapshot(src)) == ["converted/coro_cube.fits"]


def test_read_only_blocks_writes_and_restores(tmp_path):
    src = _tree(tmp_path / "src")
    target = src / "converted" / "coro_cube.fits"
    with at.read_only(src / "converted"):
        assert not os.stat(target).st_mode & stat.S_IWUSR
        if os.geteuid() != 0:
            with pytest.raises(PermissionError):
                target.write_bytes(b"z")
    assert os.stat(target).st_mode & stat.S_IWUSR

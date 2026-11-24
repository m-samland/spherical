"""Tests for `spherical.pipeline.steps.download_data`.

Covers the two robustness fixes:

* the glob widening (matching ``.fits.Z`` as well as ``.fits``) is enforced
  at the caller boundary — here we test the resolver's behaviour when fed
  both file kinds.
* `update_observation_file_paths` now raises a `FileNotFoundError` with a
  branched hint (download enabled vs. disabled) when any DP.ID cannot be
  matched on disk.
* `download_data_for_observation` unpacks leftover ``.fits.Z`` files that
  a prior interrupted download left behind.
"""
from __future__ import annotations

import gzip
import shutil
import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


def _make_observation(frames_by_key: dict[str, list[str]]) -> MagicMock:
    observation = MagicMock()
    observation.frames = {
        key: Table({"DP.ID": np.asarray(ids, dtype="U64")})
        for key, ids in frames_by_key.items()
    }
    return observation


class TestUpdateObservationFilePathsResolution:
    def test_matches_fits_and_fits_z(self, tmp_path):
        from spherical.pipeline.steps.download_data import update_observation_file_paths

        fits_path = tmp_path / "SPHER.a.fits"
        z_path = tmp_path / "SPHER.b.fits.Z"
        fits_path.write_bytes(b"")
        z_path.write_bytes(b"")

        observation = _make_observation({"CORO": ["SPHER.a", "SPHER.b"]})
        update_observation_file_paths(
            [str(fits_path), str(z_path)],
            observation,
            logger=MagicMock(),
            used_keys=("CORO",),
        )
        resolved = list(observation.frames["CORO"]["FILE"])
        assert str(fits_path) in resolved
        assert str(z_path) in resolved


class TestUpdateObservationFilePathsRaises:
    def test_raises_when_download_enabled_and_files_missing(self, tmp_path):
        from spherical.pipeline.steps.download_data import update_observation_file_paths

        observation = _make_observation({"CORO": ["SPHER.a", "SPHER.b"]})
        with pytest.raises(FileNotFoundError, match="Download step ran"):
            update_observation_file_paths(
                [],
                observation,
                logger=MagicMock(),
                used_keys=("CORO",),
                download_was_enabled=True,
            )

    def test_raises_when_download_disabled_and_files_missing(self, tmp_path):
        from spherical.pipeline.steps.download_data import update_observation_file_paths

        observation = _make_observation({"CORO": ["SPHER.a"]})
        with pytest.raises(FileNotFoundError, match="steps.download_data=True"):
            update_observation_file_paths(
                [],
                observation,
                logger=MagicMock(),
                used_keys=("CORO",),
                download_was_enabled=False,
                raw_directory=str(tmp_path),
            )

    def test_no_raise_when_all_resolved(self, tmp_path):
        from spherical.pipeline.steps.download_data import update_observation_file_paths

        p = tmp_path / "SPHER.a.fits"
        p.write_bytes(b"")
        observation = _make_observation({"CORO": ["SPHER.a"]})
        update_observation_file_paths(
            [str(p)],
            observation,
            logger=MagicMock(),
            used_keys=("CORO",),
        )
        assert observation.frames["CORO"]["FILE"][0] == str(p)


class TestDecompressLeftoverZFiles:
    def _write_gzip_fits(self, path):
        primary = fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32))
        buf = fits.HDUList([primary])
        tmp_uncompressed = path.with_suffix("")
        buf.writeto(tmp_uncompressed, overwrite=True)
        with open(tmp_uncompressed, "rb") as fin, gzip.open(path, "wb") as fout:
            fout.write(fin.read())
        tmp_uncompressed.unlink()

    def test_unpacks_leftover_z_files_and_removes_source(self, tmp_path):
        from spherical.pipeline.steps.download_data import _decompress_leftover_z_files

        z_path = tmp_path / "CORO" / "SPHER.leftover.fits.Z"
        z_path.parent.mkdir(parents=True)
        self._write_gzip_fits(z_path)

        _decompress_leftover_z_files(tmp_path, tmp_path, logger=MagicMock())

        assert not z_path.exists()
        assert (z_path.parent / "SPHER.leftover.fits").exists()

    @pytest.mark.skipif(shutil.which("compress") is None, reason="needs Unix compress to write the LZW file")
    def test_unpacks_lzw_files_without_uncompress(self, tmp_path, monkeypatch):
        from spherical.pipeline.steps.download_data import _decompress_leftover_z_files

        fits_path = tmp_path / "SPHER.lzw.fits"
        fits.PrimaryHDU(np.arange(4096, dtype=np.int16).reshape(64, 64)).writeto(fits_path)
        subprocess.run(["compress", "-f", str(fits_path)], check=True)
        monkeypatch.setenv("PATH", "")

        _decompress_leftover_z_files(tmp_path, tmp_path, logger=MagicMock())

        assert sorted(p.name for p in tmp_path.iterdir()) == ["SPHER.lzw.fits"]
        np.testing.assert_array_equal(fits.getdata(fits_path), np.arange(4096, dtype=np.int16).reshape(64, 64))

    def test_corrupt_file_leaves_no_output(self, tmp_path):
        from spherical.pipeline.steps.download_data import _decompress_leftover_z_files

        z_path = tmp_path / "SPHER.corrupt.fits.Z"
        z_path.write_bytes(gzip.compress(b"x" * 10000)[:-20])

        with pytest.raises(EOFError):
            _decompress_leftover_z_files(tmp_path, tmp_path, logger=MagicMock())

        assert sorted(tmp_path.iterdir()) == [z_path]

    def test_noop_when_no_leftover(self, tmp_path):
        from spherical.pipeline.steps.download_data import _decompress_leftover_z_files

        (tmp_path / "SPHER.a.fits").write_bytes(b"")
        _decompress_leftover_z_files(tmp_path, tmp_path, logger=MagicMock())
        assert (tmp_path / "SPHER.a.fits").exists()

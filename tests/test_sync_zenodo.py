from spherical.scripts import sync_zenodo_tables as sz


def test_wanted_filenames_all_includes_polarimetry():
    names = sz._wanted_filenames("all", include_polarimetry=True)
    assert "table_of_observations_irdis_polarimetry.fits" in names
    assert "table_of_files_ifs.csv" in names


def test_wanted_filenames_ifs_only():
    names = sz._wanted_filenames("ifs", include_polarimetry=False)
    assert all("irdis" not in n for n in names)


def test_checksum_missing_file_is_false(tmp_path):
    assert sz._checksum_matches(tmp_path / "nope.csv", "md5:abc") is False


def test_checksum_non_md5_is_unverifiable(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("hello")
    # sha256-style checksum: cannot verify, must not claim a match nor a mismatch.
    assert sz._checksum_matches(f, "sha256:deadbeef") is False
    assert sz.checksum_verifiable("sha256:deadbeef") is False


def test_checksum_md5_matches(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"hello")
    import hashlib
    md5 = hashlib.md5(b"hello").hexdigest()
    assert sz._checksum_matches(f, f"md5:{md5}") is True
    assert sz.checksum_verifiable(f"md5:{md5}") is True

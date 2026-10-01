"""Tests for the spherical metadata written into reduced FITS headers."""

from spherical.pipeline.fits import headers


def test_header_has_no_esorex_keywords(monkeypatch):
    """The reduction does not use the ESO pipeline, so no esorex version is recorded (#160)."""
    # os.getlogin raises without a controlling terminal, as on CI runners.
    monkeypatch.setattr(headers, "getlogin", lambda: "tester")
    header = headers.spherical_populate_fits_header(None)
    assert "HIERARCH SPHERICAL CHARIS VERSION" in header
    assert not [key for key in header if "ESOREX" in key]

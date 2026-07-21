"""IRDIS obs-mode → species filter-name mapping for TRAP template matching.

TRAP's ``SpectralTemplate`` uses ``species.SyntheticPhotometry(filter_name)`` to
integrate model spectra through each channel's bandpass when
``instrument_type == 'photometry'``. This module holds the SPHERE-specific
translation from IRDIS observation-mode strings (as they appear in the
observation table's ``FILTER`` column) to the SVO / species filter names.

Only dual-band imaging modes are listed. Single-channel modes (broadband,
narrow-band, dual polarimetry) are absent by design: template matching on one
channel reduces to a constant and provides no discriminating power. Callers
treat a missing entry as "not a template-matching-capable mode" and fall back
to the template-free detection path.
"""

from __future__ import annotations

IRDIS_SPECIES_FILTERS: dict[str, tuple[str, str]] = {
    "DB_K12": ("Paranal/SPHERE.IRDIS_D_K12_1", "Paranal/SPHERE.IRDIS_D_K12_2"),
    "DB_H23": ("Paranal/SPHERE.IRDIS_D_H23_2", "Paranal/SPHERE.IRDIS_D_H23_3"),
    "DB_H34": ("Paranal/SPHERE.IRDIS_D_H34_3", "Paranal/SPHERE.IRDIS_D_H34_4"),
    "DB_Y23": ("Paranal/SPHERE.IRDIS_D_Y23_2", "Paranal/SPHERE.IRDIS_D_Y23_3"),
    "DB_J23": ("Paranal/SPHERE.IRDIS_D_J23_2", "Paranal/SPHERE.IRDIS_D_J23_3"),
}


def species_filters_for_mode(obs_mode: str) -> tuple[str, str] | None:
    """Return the (left, right) species filter names for an IRDIS obs mode.

    Parameters
    ----------
    obs_mode
        IRDIS observation-mode string (e.g. ``"DB_K12"``).

    Returns
    -------
    tuple of (str, str) or None
        Two SVO-style species filter names for the left/right detector halves
        of a dual-band mode, or ``None`` for any mode not in the mapping.
    """
    return IRDIS_SPECIES_FILTERS.get(obs_mode)

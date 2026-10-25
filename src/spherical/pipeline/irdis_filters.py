"""IRDIS obs-mode → species filter-name mapping for TRAP template matching.

TRAP's ``SpectralTemplate`` uses ``species.SyntheticPhotometry(filter_name)`` to
integrate model spectra through each channel's bandpass when
``instrument_type == 'photometry'``. This module holds the SPHERE-specific
translation from IRDIS observation-mode strings (as they appear in the
observation table's ``FILTER`` column) to the SVO / species filter names, one
per detector half.

A mode listed here runs template matching. The pair also decides which
templates mean anything: dual-band modes carry two distinct bandpasses and fit
all templates, while broadband carries the same filter on both halves. There a
flat template is the optimal two-channel collapse for detection, but it cannot
tell spectral types apart, so only ``"flat"`` is fitted (see
:func:`default_template_names_for_mode`).

Narrow-band (``NB_*``) and dual polarimetry (``DP_*``) are not mapped yet.
Callers treat a missing entry as "not a template-matching-capable mode" and
fall back to the template-free detection path.
"""

from __future__ import annotations

IRDIS_SPECIES_FILTERS: dict[str, tuple[str, str]] = {
    "DB_K12": ("Paranal/SPHERE.IRDIS_D_K12_1", "Paranal/SPHERE.IRDIS_D_K12_2"),
    "DB_H23": ("Paranal/SPHERE.IRDIS_D_H23_2", "Paranal/SPHERE.IRDIS_D_H23_3"),
    # Same D_H23 dual wheel behind the B_ND-H common filter, but SVO publishes the
    # attenuated pair separately, so it does not reuse the DB_H23 names.
    "DB_NDH23": ("Paranal/SPHERE.IRDIS_D_ND-H23_2", "Paranal/SPHERE.IRDIS_D_ND-H23_3"),
    "DB_H34": ("Paranal/SPHERE.IRDIS_D_H34_3", "Paranal/SPHERE.IRDIS_D_H34_4"),
    "DB_Y23": ("Paranal/SPHERE.IRDIS_D_Y23_2", "Paranal/SPHERE.IRDIS_D_Y23_3"),
    "DB_J23": ("Paranal/SPHERE.IRDIS_D_J23_2", "Paranal/SPHERE.IRDIS_D_J23_3"),
    "BB_Y": ("Paranal/SPHERE.IRDIS_B_Y", "Paranal/SPHERE.IRDIS_B_Y"),
    "BB_J": ("Paranal/SPHERE.IRDIS_B_J", "Paranal/SPHERE.IRDIS_B_J"),
    "BB_H": ("Paranal/SPHERE.IRDIS_B_H", "Paranal/SPHERE.IRDIS_B_H"),
    "BB_Ks": ("Paranal/SPHERE.IRDIS_B_Ks", "Paranal/SPHERE.IRDIS_B_Ks"),
}

SINGLE_FILTER_TEMPLATE_NAMES: tuple[str, ...] = ("flat",)


def species_filters_for_mode(obs_mode: str) -> tuple[str, str] | None:
    """Return the (left, right) species filter names for an IRDIS obs mode.

    Parameters
    ----------
    obs_mode
        IRDIS observation-mode string (e.g. ``"DB_K12"``).

    Returns
    -------
    tuple of str or None
        SVO-style species filter names for the left/right detector halves,
        identical for a single-filter mode, or ``None`` for any mode not in
        the mapping.
    """
    return IRDIS_SPECIES_FILTERS.get(obs_mode)


def is_single_filter_mode(obs_mode: str) -> bool:
    """Return whether both detector halves of *obs_mode* carry the same filter.

    Unmapped modes return ``False``: they never reach template matching.
    """
    filters = species_filters_for_mode(obs_mode)
    return filters is not None and len(set(filters)) == 1


SINGLE_FILTER_OBS_MODES: frozenset[str] = frozenset(
    mode for mode in IRDIS_SPECIES_FILTERS if is_single_filter_mode(mode)
)


def default_template_names_for_mode(obs_mode: str) -> tuple[str, ...] | None:
    """Return the TRAP templates worth fitting for *obs_mode*.

    Parameters
    ----------
    obs_mode
        IRDIS observation-mode string.

    Returns
    -------
    tuple of str or None
        ``("flat",)`` for a single-filter mode, where L-type duplicates flat
        and the T-type offset fit is singular; ``None`` (all templates) for
        every other mode.
    """
    if is_single_filter_mode(obs_mode):
        return SINGLE_FILTER_TEMPLATE_NAMES
    return None

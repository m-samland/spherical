# IRDIS template-matching detection — design

**Status:** approved, ready for planning
**Date:** 2026-07-20
**Scope:** enable TRAP's template-matching detection for IRDIS dual-band imaging (DBI) modes; fall back gracefully to `detection_and_characterization()` for single-channel modes.

## Problem

TRAP's template-matching detection currently crashes for IRDIS observations. From
the 51 Eri DB_K12 2015-09-24 run:

```
File "trap/src/trap/template.py", line 209, in __init__
    self.mean_normalized_contrast_value = np.mean(self.contrast_modelbox.flux)
AttributeError: 'SpectralTemplate' object has no attribute 'contrast_modelbox'
```

Root cause: `trap_config_for_irdis()` sets `instrument_type="imaging"`, which
matches neither the `'ifu'` nor the `'photometry'` branch in
`SpectralTemplate.__init__`. `contrast_modelbox` is only assigned inside one of
those branches, so the subsequent `np.mean(...)` raises. Additionally, the
`InstrumentConfig.to_instrument()` factory hard-codes `filters=None`, so even
with the correct `instrument_type` there would be nothing for TRAP's photometry
branch to integrate the model spectra through.

## Goal

Template matching works for IRDIS DBI (2-channel) observations, and single-channel
IRDIS modes (BB, NB, DP) run the template-free detection path instead of raising.

Non-goals:
- Template matching on single-channel modes (mathematically degenerate).
- Modifying TRAP's `SpectralTemplate` or `add_default_templates` — the existing
  `'photometry'` branch already computes filter-integrated contrast per channel
  via `species.SyntheticPhotometry`.
- Any changes to the IFS path.

## Architecture

The TRAP `Instrument` object is the interface between the pipeline and template
matching. Everything template matching needs — channel count, wavelengths, filter
bandpasses — travels on that object. SPHERE-specific mapping (obs mode → species
filter names) lives in spherical; TRAP stays instrument-agnostic on this axis.

```
spherical/pipeline/irdis_filters.py       # new: {DB_K12: (D_K1, D_K2), ...}
        │
        ▼
spherical/pipeline/run_trap.py             # after to_instrument(), inject
        │                                    used_instrument.filters
        ▼
TRAP Instrument (filters=[…], wavelengths=[…], instrument_type='photometry')
        │
        ▼
TRAP SpectralTemplate 'photometry' branch  # unchanged; already integrates
        │                                    model spectra through species filters
        ▼
detection_and_characterization_with_template_matching (unchanged)
```

For unsupported modes the pipeline swaps in
`detection_and_characterization()` (detection.py:4132), which still produces
contrast curves, normalized detection map, candidate tables, and extracted
candidate spectra — everything except the template-matching-specific outputs.

## TRAP-side changes (`../trap/src/trap/parameters.py`)

Two edits only:

1. **Line 1275, `trap_config_for_irdis()`:** rename the type so it matches
   `SpectralTemplate`'s branch labels.
   ```python
   -    instrument_type="imaging",
   +    instrument_type="photometry",
   ```
2. **Line 626, `InstrumentConfig.to_instrument()`:** accept an optional
   `filters` argument and forward it to `Instrument(...)`. Default `None`;
   existing IFS callers are unaffected.
   ```python
   def to_instrument(self, obs_mode, wavelengths=None, filters=None):
       ...
       return Instrument(..., wavelengths=wavelengths, filters=filters, ...)
   ```

**Explicitly not changed:**
- `SpectralTemplate.__init__` — its `'photometry'` branch (template.py:173-204)
  already does the right thing.
- `add_default_templates` — its `len(instrument.wavelengths) <= 2` guard
  (detection.py:3158-3164) already disables T-type slope fitting for DBI.
- Detection wrapper signatures.

**Known limitation (noted, not fixed here):** the photometry branch hard-codes
`spec_res=55` and `wavel_resample=np.linspace(0.9, 2.4, 200)` (template.py:181-186).
Fine for J/H/K IRDIS filters; will need revisiting if we ever add narrow-band or
Y-band modes to template matching.

## Spherical-side changes

### New: `src/spherical/pipeline/irdis_filters.py`

```python
"""IRDIS obs-mode → species filter-name mapping for TRAP template matching."""

IRDIS_SPECIES_FILTERS: dict[str, tuple[str, str]] = {
    "DB_K12": ("SPHERE/IRDIS.D_K1", "SPHERE/IRDIS.D_K2"),
    "DB_H23": ("SPHERE/IRDIS.D_H2", "SPHERE/IRDIS.D_H3"),
    "DB_H34": ("SPHERE/IRDIS.D_H3", "SPHERE/IRDIS.D_H4"),
    "DB_Y23": ("SPHERE/IRDIS.D_Y2", "SPHERE/IRDIS.D_Y3"),
    "DB_J23": ("SPHERE/IRDIS.D_J2", "SPHERE/IRDIS.D_J3"),
}


def species_filters_for_mode(obs_mode: str) -> tuple[str, str] | None:
    """Return (left, right) species filter names for a DBI obs mode, or None.

    None for any mode not in the mapping (BB, NB, DP, unknown) — callers use this
    as the signal to run the template-free detection path.
    """
    return IRDIS_SPECIES_FILTERS.get(obs_mode)
```

Module imports only stdlib types (no species/trap at module scope), so the
"`import spherical` works without the pipeline extra" invariant holds.

Exact species filter strings will be verified against species/SVO in Task 1 of
the implementation plan; the SVO SPHERE/IRDIS naming convention above is the
starting point.

### Edit: `src/spherical/pipeline/run_trap.py`

Two related pieces of logic to add, right around the existing
`used_instrument.wavelengths = wavelengths` block (~line 456) and the detection
call site (~line 680).

**A) Populate `used_instrument.filters` and choose the detection branch:**

```python
from spherical.pipeline.irdis_filters import species_filters_for_mode

templates_supported = True
if used_instrument.instrument_type == "photometry":
    filters = species_filters_for_mode(obs_mode)
    if filters is None:
        templates_supported = False
        logger.info(
            f"Template matching not applicable for {obs_mode} "
            "(single-channel or no species mapping); "
            "falling back to detection_and_characterization()."
        )
    else:
        used_instrument.filters = list(filters)
```

`obs_mode` is already available in `run_trap` (it drives `to_instrument()` and
the wavelengths / PSF paths); implementation will thread the exact variable name
in place.

**B) Detection call site — swap methods based on `templates_supported`:**

```python
if templates_supported:
    analysis.detection_and_characterization_with_template_matching(
        reduction_parameters=deepcopy(analysis.reduction_parameters),
        instrument=analysis.instrument,
        species_database_directory=species_database_directory,
        stellar_parameters=trap_config.get_stellar_parameters(),
        # ... all existing kwargs unchanged ...
    )
else:
    analysis.detection_and_characterization(
        data_full=data_full, flux_psf_full=flux_psf_full, pa=pa,
        temporal_components_fraction=trap_config.processing.temporal_components_fraction[0],
        inverse_variance_full=inverse_variance_full,
        bad_frames=bad_frames, bad_pixel_mask_full=bad_pixel_mask_full,
        xy_image_centers=xy_image_centers,
        amplitude_modulation_full=amplitude_modulation_full,
        candidate_threshold=trap_config.detection.candidate_threshold,
        detection_threshold=trap_config.detection.detection_threshold,
        search_radius=trap_config.detection.search_radius,
        good_fraction_threshold=trap_config.detection.good_fraction_threshold,
        theta_deviation_threshold=trap_config.detection.theta_deviation_threshold,
        yx_fwhm_ratio_threshold=trap_config.detection.yx_fwhm_ratio_threshold,
        save_initial_detection_products=trap_config.detection.save_initial_detection_products,
    )
```

Selection rules:

| `instrument_type` | mapping | detection path |
|---|---|---|
| `'ifu'` | n/a | `detection_and_characterization_with_template_matching` (IFS, unchanged) |
| `'photometry'` | present (DB mode) | `detection_and_characterization_with_template_matching` |
| `'photometry'` | absent (BB/NB/DP/unknown) | `detection_and_characterization` (fallback) |

Any error resolving filters (e.g. species can't find `SPHERE/IRDIS.D_K1`) is
caught, logged, and downgraded to the fallback so the observation still yields
science products.

## Verification & testing

### Unit tests — `tests/pipeline/test_irdis_template.py` (new)

1. **`test_species_filters_for_mode_dbi`** — every DB mode in
   `IRDIS_SPECIES_FILTERS` returns a 2-tuple of non-empty strings; BB / NB /
   unknown modes return `None`.
2. **`test_spectral_template_photometry_branch_smoke`** — construct a
   `trap.parameters.Instrument` with `instrument_type='photometry'`,
   `wavelengths=[2.11, 2.25] * u.micron`, `filters=['SPHERE/IRDIS.D_K1',
   'SPHERE/IRDIS.D_K2']`, feed it two tiny synthetic model boxes, assert
   `SpectralTemplate` builds and `.contrast_modelbox.flux` has length 2.
   Guarded with `pytest.importorskip("species")` and a try/except around the
   `SyntheticPhotometry` call so it skips cleanly when species DB isn't
   initialized (CI environment, per CLAUDE.md).

### Manual end-to-end

- **51 Eri DB_K12 2015-09-24** — the case that currently crashes. Success
  criteria: `trap_reduction.log` shows no `AttributeError`;
  `template_matching_detection` products (L-type / T-type / flat maps) land in
  the result folder.
- **One BB_H target** (any available) — confirm the fallback branch produces
  `companion_table.csv`, contrast curve PDF, and normalized detection cube.

### Guarantee to keep

`python -c "import spherical"` continues to work without the pipeline extra.
The new module is stdlib-only; `run_trap.py` (which imports it) is already
pipeline-extra territory.

## Out of scope

- BB / NB filter mappings (single-channel template matching not scientifically
  meaningful).
- Modifying `SpectralTemplate.__init__` — existing photometry branch is
  sufficient.
- Modifying `add_default_templates`.
- Bootstrap of SPHERE/IRDIS filters into the species DB on a completely empty
  install — first-run verification will tell us whether species auto-fetches
  from SVO; if not, that becomes a small follow-up.
- Transmission-correction path (`correct_transmission=True` in
  `SpectralTemplate`) — still `NotImplementedError` upstream; unrelated to this
  fix.

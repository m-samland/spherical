# TRAP settings

The companion search runs in [TRAP](https://github.com/m-samland/trap)
([Samland et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..24S/abstract)),
which has its own configuration object. The reduction templates build it with
`trap_config_for_ifs()` or `trap_config_for_irdis()` and then change a few values.
TRAP has no hosted documentation yet, so this page lists the settings that matter
when you reduce SPHERE data with spherical.

You change a TRAP setting the same way as a spherical one, by replacing a part of the
configuration with a modified copy:

```python
trap_config.reduction = trap_config.reduction.merge(search_region_outer_bound=65)
```

## Instrument presets

These values come from the two preset functions. The SPHERE-specific ones
(`right_handed`, `yx_anamorphism`) should stay as they are.

| Setting | IFS | IRDIS | Meaning |
|---|---|---|---|
| `reduction.search_region_inner_bound` | 1 | 1 | Innermost radius of the reduced region, in pixels. The coronagraph's inner working angle is handled by the transmission curve, not by this bound. |
| `reduction.search_region_outer_bound` | 81 | 200 | Outermost radius, in pixels. |
| `reduction.temporal_model` | True | True | Model the stellar speckles with temporal principal components. |
| `reduction.spatial_model` | False | False | Spatial model, not used for SPHERE. |
| `reduction.right_handed` | False | False | Sense of the derotation angle for SPHERE data. Do not change. |
| `reduction.yx_anamorphism` | [1.0059, 1.0011] | [1.0062, 1.0] | Anamorphism correction applied in the forward model. Set to [1.0, 1.0] only if the images are already corrected (see `irdis_preprocessing.correct_anamorphism`). |
| `reduction.auto_footprint` | True | True | Work out the usable detector footprint from the data. |
| `processing.wavelength_indices` | range(1, 38) | range(0, 2) | Wavelength channels to reduce. IFS skips the first and last channel. |
| `instrument.pixel_scale_arcsec_per_pixel` | 0.00746 | 0.01225 | Pixel scale in arcseconds. |
| `instrument.instrument_type` | ifu | photometry | IRDIS integrates model spectra through the two filters for template matching. |

## Settings you choose

The templates set or suggest these values. The search region has the largest
effect on run time: the cost grows with the area between the inner and outer bound.

| Setting | IFS template | IRDIS template | How to choose |
|---|---|---|---|
| `reduction.search_region_outer_bound` | 65 | 200 (preset) | Large enough to cover the separations you want to search. |
| `detection.detection_threshold` | 5.0 | 5.0 (suggested) | Signal-to-noise above which a source counts as detected. |
| `detection.candidate_threshold` | 4.75 | 4.75 (suggested) | Lower threshold for candidates that are then characterised. |
| `detection.search_radius` | 15 px | 11 px (suggested) | Radius for matching the same source across templates and channels. |
| `detection.use_spectral_correlation` | False | False (suggested) | Keep False for IRDIS, which has only two channels. |
| `processing.temporal_components_fraction` | [0.15] | [0.2] (suggested) | Fraction of temporal principal components in the speckle model. A list runs one reduction per value. |
| `reduction.scratch_dir` | unset | unset | Where TRAP stores data shared with its worker processes. Unset, TRAP uses `/dev/shm` if it exists and has room, otherwise the system temporary directory. On shared servers, point it at disk. |

## Settings filled in by spherical

`run_trap_on_observations` fills these TRAP inputs from the reduction products,
controlled by fields of the reduction configuration (see
[Configuration](configuration.md)).

| spherical setting | TRAP input it fills |
|---|---|
| `use_gaia_stellar_parameters` | Stellar parameters for template matching (Gaia DR3, then spectral type). |
| `apply_coronagraph_transmission` | `coronagraph_transmission`, the packaged transmission curve. |
| `pass_inverse_variance_to_trap` | `inverse_variance_full`, noise weights from the inverse-variance cubes. |
| `derive_trap_bad_pixels_from_ivar`, `ivar_bad_pixel_ratio_threshold`, `ivar_bad_pixel_frame_fraction` | `bad_pixel_mask_full`, when no calibration bad-pixel map exists. |
| `pass_amplitude_modulation_to_trap` | `amplitude_modulation_full`, for continuous-waffle sequences. |
| `pass_center_outliers_as_bad_frames_to_trap` | `bad_frames`, for continuous-waffle sequences. |

## CPU count

TRAP gets its CPU count only from `config.apply_trap_resources(trap_config)`. Call it
right after creating `trap_config` and before any `trap_config.reduction.merge(...)`,
because it resets `reduction.scratch_dir`.

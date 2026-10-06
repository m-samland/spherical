# Output products

After reading this page you can find every file a reduction writes and know which ones
to use for science. Axis order, units, centre and rotation of the files are on
[Conventions](../reference/conventions.md). The examples come from the 51 Eri sequence
of 2015-09-24.

## Where things go

A reduction writes raw data under `config.directories.raw_directory`, and reduction
products and TRAP results under `config.directories.reduction_directory`. `{target}` is the SIMBAD name with spaces replaced by `_`, for
example `*_51_Eri`.

::::{tab-set}
:sync-group: instrument

:::{tab-item} IFS
:sync: ifs

```text
raw_directory/IFS/
├── science/{target}/{filter}/{date}/
│   ├── CORO/  CENTER/  FLUX/        raw frames as downloaded from ESO
└── calibration/{filter}/WAVECAL/    raw wavelength calibration frames

reduction_directory/IFS/
├── calibration/{filter}/{time}/     charis wavelength solution
├── observation/{target}/{filter}/{date}/
│   ├── reduction.log                human-readable log of the latest run
│   ├── reduction.jsonlog            the same as JSON lines, for reduction_status
│   ├── old_logs/                    logs of earlier runs, with a time stamp
│   ├── crash_report.txt             only if the reduction failed
│   └── optext/                      one folder per charis extraction method
│       ├── CORO/  CENTER/  FLUX/    one extracted cube per raw frame
│       ├── .extract_cubes.done      completion marker
│       └── converted/               the products, see below
└── trap/{target}/{filter}/{date}/   TRAP results, see below
```
:::

:::{tab-item} IRDIS
:sync: irdis

```text
raw_directory/IRDIS/
├── science/{target}/{filter}/{date}/
│   ├── CORO/  CENTER/  FLUX/        raw frames as downloaded from ESO
└── calibration/{filter}/
    ├── FLAT/  BG_SCIENCE/           raw flats and background frames

reduction_directory/IRDIS/
├── calibration/{filter}/{date}/     master background, flat and bad-pixel map
├── observation/{target}/{filter}/{date}/
│   ├── reduction.log                human-readable log of the latest run
│   ├── reduction.jsonlog            the same as JSON lines, for reduction_status
│   ├── old_logs/                    logs of earlier runs, with a time stamp
│   ├── crash_report.txt             only if the reduction failed
│   └── converted/                   the products, see below
└── trap/{target}/{filter}/{date}/   TRAP results, see below
```
:::
::::

## The products folder

`converted/` holds the same files for both instruments, except where noted.
`{type}` is `coro`, `center` or `flux`.

```text
converted/
├── {type}_cube.fits                       data cube per frame type
├── {type}_ivar_cube.fits                  inverse variance of each cube
├── frames_info_{type}.csv                 one row per frame: times, PARANG, DEROT ANGLE, header values
├── {type}_parallactic_angles.fits         IFS only: DEROT ANGLE per frame
├── wavelengths.fits                       channel wavelengths in nm
├── badpixel_map.fits                      IRDIS only: bad-pixel map per channel
├── image_centers.fits                     star position fitted in each CENTER frame
├── image_centers_fitted.fits              intermediate step of the clean-up
├── image_centers_fitted_robust.fits       star positions used by TRAP (see the note below)
├── center_plots/                          plots of the centre fits and their evolution
├── psf_cube_for_postprocessing.fits       stellar PSF from the FLUX frames, used by TRAP
├── psf_cube_for_postprocessing_unrepaired.fits  the PSF before repairing bad pixels, for checks
├── flux_amplitude_calibrated.fits         FLUX-frame photometry
├── flux_calibration_indices.csv           which FLUX block calibrates which science frames
├── spot_amplitude_variation.fits          stellar flux variation from the waffle spots
├── flux_plots/                            plots of the flux calibration
├── additional_outputs/                    intermediate measurements: spot and PSF stamps,
│                                          fitted amplitudes, ND attenuation, signal-to-noise
├── coro_cube_aligned.fits                 only with align_frames: the science cube with the star
│                                          on the central pixel (center_cube_aligned.fits for
│                                          continuous waffle), no inverse variance
└── .align_frames.done                     only with align_frames: completion marker
```

`image_centers_fitted_robust.fits` has one entry per science frame for IRDIS and for
continuous-waffle IFS sequences. For other IFS sequences it has one entry per
CENTER frame, (39, 4, 2) for 51 Eri. TRAP averages them over the sequence and uses
that position, per wavelength, for every science frame
(`science_frames.normalize_centers_to_frames`).

For 51 Eri the IFS `coro_cube.fits` has shape (39, 256, 262, 262) and the IRDIS
`DB_K12` one (2, 256, 1024, 1024). The IFS `psf_cube_for_postprocessing.fits` has
shape (39, 2, 57, 57), one PSF per wavelength and FLUX block. Turning on
`bundle_hexagons` or `bundle_residuals` adds `{type}_hexagons_cube.fits` or
`{type}_residuals_cube.fits` for IFS.

## The TRAP folder

TRAP names its files after the run settings. `ncomp038_frac0.15` means 38 principal
components from `temporal_components_fraction=0.15`, and `temporal` names the model.
One run writes these files.

```text
trap/{target}/{filter}/{date}/
├── detection_lam{NN}_{run}.fits           detection map of channel NN, shape (3, ny, nx)
├── detection_{run}.fits                   all channels, shape (channels, 3, ny, nx)
├── norm_detection_{run}.fits              signal-to-noise normalised per separation
├── uncertainty_image_{run}.fits           contrast uncertainty
├── median_uncertainty_image_{run}.fits    its median at each separation
├── contrast_table_{run}.csv  (.obj)       detection limits per channel and separation
├── reduction_config.obj, instrument.obj   TRAP's settings for this run (Python pickles)
├── template_matching/                     results combined with spectral templates
├── trap_reduction.log, .jsonlog           TRAP log of the latest run
├── trap_crash_report.txt                  only if TRAP failed
└── .run_trap_detection.done               completion marker
```

The three planes of a {term}`Detection map` are the measured {term}`Contrast`, its
uncertainty, and the signal-to-noise, which is the first divided by the second
(`trap/reduction_wrapper.py`, `fill_detection_image`). For 51 Eri the IFS maps are
195 × 195 pixels for 37 channels, since the IFS preset leaves out the first and last
channel, and the IRDIS maps 163 × 163 for 2 channels.

The contrast tables have one row per channel and separation, with the columns
`wavelength_index`, `sep (pix)`, `sep (mas)`, the minimum contrast and the percentiles
`contrast_0.15` to `contrast_99.85`, and `snr_normalization`.

`template_matching/` holds, for each template (`flat`, `L-type`, `T-type`), the
contrast table, uncertainty maps and normalised detection map, and these candidate
tables.

`companion_table_{template}.csv`, `validated_companion_table_{template}.csv`
: Position and fit of each candidate, before and after validation.

`validated_companion_table_short_{template}.csv`
: One row per detected companion and channel, with `separation`, `position_angle`,
  `wavelength`, `contrast` and `uncertainty`. This is the spectrum and astrometry of a
  companion.

`overall_companion_detections.csv`, `overall_validated_companion_detections.csv`
: The candidates combined over the templates, with the template that fits best. The
  `_spectra` variants add one row per channel.

`per_channel_astrometry.csv`
: The fitted position of each detection in every channel.

The plots `contrast_plot_*.png` (and PDF) and `companion_spectra_*.pdf` show the same
results. IRDIS observations with a single filter skip template matching
(`run_trap.py`).

```{common-mistake}
TRAP's tables give `wavelength` in µm (2.11 for the first `DB_K12` channel), while
spherical's `wavelengths.fits` is in nm (2110). `separation` and the relative
positions `x_relative` and `y_relative` are in pixels, and `position_angle` is in
degrees east of north.
```

## Files for science

| You want | Read |
|---|---|
| The spectrum or photometry and position of a companion | `template_matching/validated_companion_table_short_{template}.csv` |
| Detection limits | `contrast_table_{run}.csv`, or the one in `template_matching/` |
| A map to look for companions | `norm_detection_{run}.fits` |
| Data for your own post-processing | `coro_cube.fits`, `frames_info_coro.csv`, `image_centers_fitted_robust.fits` and `psf_cube_for_postprocessing.fits`, or `coro_cube_aligned.fits` |

For your own post-processing, read [Conventions](../reference/conventions.md) first.
The angle file `*_parallactic_angles.fits` holds `DEROT ANGLE`, not the parallactic
angle.

## Removing intermediate files

The raw frames, the extracted cubes of single frames and the charis calibrations are
not needed once a reduction is finished. `cleanup_pipeline_products` deletes them for
IFS reductions after checking that the bundled cubes exist. Run it with
`dry_run=True` first to see what it would delete. It looks for the products in
`{method}/converted/` and so does not recognise IRDIS reductions.

Next: [Coming from another pipeline or instrument](other-pipelines.md)

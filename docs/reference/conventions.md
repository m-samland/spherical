# Conventions

After reading this page you can read spherical's products correctly: where the star
is, which way the field is rotated, and the units and axis order of each file. Each
entry names the code that defines it.

## Star centre

After the optional `align_frames` step, the star sits on pixel `N // 2` in both
axes, counted from 0. {term}`IFS` cubes leave {term}`charis` at 262 × 262 pixels and
are padded to 263, so pixel 131 is both `N // 2` and the geometric centre
(`steps/align_frames.py`, `pad_to_odd` and `shift_to_target`). TRAP makes the same
assumption.

Cubes that are not aligned keep the star where it fell. Its measured position is in
`converted/image_centers_fitted_robust.fits`, one `(x, y)` pair per wavelength and
frame, shape `(wavelength, frame, 2)`, in pixels counted from 0. `run_trap.py`
passes the file to TRAP, which swaps each pair to `(y, x)` itself. For an IFS sequence without
continuous {term}`Waffle spots` the file holds one entry per {term}`CENTER frame`,
and `normalize_centers_to_frames` spreads it over the science frames.

## Rotation angle

spherical computes one angle per frame,

```text
DEROT ANGLE = PARANG + pupil offset + instrument offset + true north
```

in `database/metadata.py`, `compute_angles`. The pupil offset is +135.99° in the
`ELEV` derotator mode ({term}`Pupil tracking`), the instrument offset is -100.48°
for IFS and 0° for {term}`IRDIS`, and {term}`True north` is -1.75°. `PARANG` and
`DEROT ANGLE` are computed at the middle of each exposure (see
[Time stamps](#time-stamps)).

```{instrument-background}
In pupil-tracking mode the derotator holds the telescope pupil, and with it the
star's speckle pattern, fixed on the detector, so the sky turns during the night.
That rotation is what {term}`ADI` uses. See the
[ESO SPHERE User Manual](https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc.html)
for the derotator modes.
```

On the detector, displayed with the origin at the lower left
(`imshow(..., origin="lower")`), a source appears rotated **clockwise** by
`DEROT ANGLE` about the star, compared with its position in a North-up, East-left
image. To put North up, rotate the frame **counter-clockwise** by `DEROT ANGLE`.
With scipy that is

```python
from scipy import ndimage

north_up = ndimage.rotate(frame, -derot_angle, reshape=False)
```

This rotates about the array centre, so it is exact about the star only when the
star is on `N // 2` of an odd-sized frame, as it is after `align_frames`.

spherical never rotates images. `align_frames` only shifts them, and TRAP moves its
companion model through the frames instead of moving the data
(`trap/makesource.py`, `yx_position_in_cube`, with `right_handed=False` for SPHERE).
The rule above is how TRAP uses `DEROT ANGLE`, and
`tests/pipeline/test_rotation_convention.py` pins it, including the scipy line. The
regression tests `tests/regression/test_51eri_astrometry_regression.py` (IRDIS) and
`test_51eri_ifs_astrometry_regression.py` (IFS) check the resulting astrometry of
51 Eri b against GRAVITY.

```{common-mistake}
`*_parallactic_angles.fits` (written by `steps/bundle_output.py`) holds
`DEROT ANGLE` for each frame, not the {term}`Parallactic angle`. If you pass it to
another pipeline as `PARANG`, the offsets are applied twice.
```

## Position angle

TRAP reports position angles east of north, counted counter-clockwise in a North-up,
East-left image (`trap/image_coordinates.py`, `relative_yx_to_rhophi`).

## Pixel scale and anamorphism

IRDIS has 12.25 mas per pixel and IFS 7.46 mas per pixel (`steps/find_star.py`,
`steps/flux_psf_calibration.py`).

SPHERE's optics stretch the image slightly along one detector axis
({term}`Anamorphism`). spherical leaves the data as they are and TRAP corrects the
stretch in its forward model, with `yx_anamorphism = [1.0059, 1.0011]` for IFS and
`[1.0062, 1.0]` for IRDIS (`trap/parameters.py`). For IRDIS you can instead correct
the images during pre-processing with `IRDISPreprocessConfig.correct_anamorphism`,
which is off by default. If you turn it on, set TRAP's `yx_anamorphism` to
`[1.0, 1.0]`, or the correction is applied twice. The IRDIS cube headers record the
choice in `SPHERICAL ANAMORPHISM APPLIED` and `SPHERICAL ANAMORPHISM FACTOR`.

## Axis order

All files are in `{reduction_directory}/{INSTRUMENT}/observation/{target}/{filter}/{date}/`,
under `{method}/converted/` for IFS and `converted/` for IRDIS (see
[On-disk layout](#on-disk-layout)). `{type}` is `coro`, `center` or `flux`.

| File | Axes | Instrument | Written by |
|---|---|---|---|
| `{type}_cube.fits`, `{type}_ivar_cube.fits` | (wavelength, frame, y, x) | both | `steps/bundle_output.py` (IFS), `steps/irdis_preprocess.py` (IRDIS) |
| `coro_cube_aligned.fits`, `center_cube_aligned.fits` | (wavelength, frame, y, x) | both | `steps/align_frames.py` |
| `{type}_parallactic_angles.fits` | (frame) | IFS | `steps/bundle_output.py` |
| `image_centers_fitted_robust.fits` | (wavelength, frame, 2), pairs `(x, y)` | both | `steps/process_centers.py` |
| `wavelengths.fits` | (wavelength) | both | `steps/bundle_output.py` (IFS), `steps/irdis_preprocess.py` (IRDIS) |
| `psf_cube_for_postprocessing.fits` | (wavelength, flux block, y, x) | both | `steps/flux_psf_calibration.py` |

IRDIS writes no angle file; its angles are in the `DEROT ANGLE` column of
`frames_info_*.csv`, which is also where TRAP reads them for both instruments.

In the 51 Eri example of 2015-09-24 the IFS `coro_cube.fits` has shape
(39, 256, 262, 262) and the IRDIS `DB_K12` one (2, 256, 1024, 1024).

## Units

Wavelengths
: Nanometres. The IFS `OBS_H` channels run from 920 to 1700 nm. The IRDIS `DB_K12`
  channels are 2110 and 2251 nm.

Data cubes
: Detector counts per exposure (DIT), not divided by the exposure time. charis
  extracts the IFS cubes from the raw counts.

Flux calibration
: The {term}`FLUX frame`s have their own DIT and neutral-density filter.
  `steps/flux_psf_calibration.py` scales them to the most common DIT of the
  {term}`CENTER frame`s and divides out the filter's attenuation, so the stellar
  PSF is on the same scale as the science frames.

Contrast
: TRAP's {term}`Contrast` is the ratio of companion flux to stellar flux, without
  units.

(time-stamps)=
## Time stamps

The frames table records three times for each frame, `TIME START`, `TIME` and
`TIME END`, with matching `MJD START`, `MJD` and `MJD END`
(`database/metadata.py`, `compute_times`). `TIME START` is the start of the
exposure including `DET DITDELAY`, `TIME` is the middle (start plus half the DIT)
and `TIME END` is start plus DIT. `PARANG` and `DEROT ANGLE` use `TIME`.

(on-disk-layout)=
## On-disk layout

Reductions write to

```text
{reduction_directory}/{INSTRUMENT}/observation/{target}/{filter}/{date}
```

(`ifs_reduction.py` and `irdis_reduction.py`). IFS products sit one level further
down, in `{method}/converted`, where `{method}` is the charis extraction method
(`cleanup.py`). TRAP results go to

```text
{reduction_directory}/{instrument}/trap/{target}/{filter}/{date}
```

(`step_registry.trap_result_folder`). `{target}` is the SIMBAD main identifier with
its spaces replaced by `_` (`step_registry.target_folder_string`), and it keeps
SIMBAD's type prefix. For the 51 Eri example the IFS products are in

```text
IFS/observation/*_51_Eri/OBS_H/2015-09-24/optext/converted
```

## FITS keywords

spherical adds these keywords to the headers of the cubes it writes, in
`pipeline/fits/headers.py` and, for the IRDIS pre-processing keywords,
`steps/irdis_preprocess.py`.

| Keywords | Content |
|---|---|
| `SPHERICAL DESC`, `SPHERICAL AUTHOR *`, `SPHERICAL PUB *` | the pipeline and the paper to cite |
| `SPHERICAL POST_PIPE VERSION`, `... GIT URL`, `... GIT HASH`, `... GIT BRANCH` | the spherical version that wrote the file |
| `SPHERICAL POST_PIPE CHARIS VERSION`, `... PYTHON VERSION`, `... ESOREX VERSION` | the software around it |
| `SPHERICAL POST_PIPE FITS AUTHOR`, `... HOSTNAME`, `... FQDN`, `... WRITE DATE`, `... WRITE TIME` | who wrote the file, where and when |
| `SPHERICAL ANAMORPHISM APPLIED`, `SPHERICAL ANAMORPHISM FACTOR`, `SPHERICAL CROP APPLIED` | IRDIS pre-processing choices |
| `SPHERICAL FRAME_INFO_FILE` and the `HIERARCH` frame keywords | the frames table and the values that are constant over the sequence |

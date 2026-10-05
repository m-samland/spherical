# Coming from another pipeline or instrument

After reading this page you can map what you know from another SPHERE pipeline, a
post-processing package or another instrument onto spherical.

## From vlt-sphere

[vlt-sphere](https://github.com/avigan/SPHERE)
([Vigan 2020](https://ascl.net/2009.002)) reduces IRDIS and IFS data into combined
cubes per frame type. spherical builds on it for parts of the IRDIS reduction and for
the astrometric and photometric calibration, so many quantities carry over.

| vlt-sphere | spherical |
|---|---|
| `science_cube.fits`, `starcenter_cube.fits`, `psf_cube.fits` | `coro_cube.fits`, `center_cube.fits`, `flux_cube.fits` in `converted/` |
| `science_derot.fits` | `DEROT ANGLE` in `frames_info_coro.csv`, and for IFS `coro_parallactic_angles.fits` |
| `science_frames.csv` | `frames_info_coro.csv` |
| frames centred on `dim // 2` when combined | star positions in `image_centers_fitted_robust.fits`; centred copy only from `align_frames` |
| anamorphism corrected when combining (default) | not corrected; TRAP corrects it in its model |
| IFS calibration and cubes from the ESO pipeline (`esorex`) | IFS cubes from {term}`charis` |

The derotation angle is the same quantity in both. It is the parallactic angle plus the
pupil offset, the instrument offset and the -1.75° {term}`True north` correction
(vlt-sphere `IFS.py`, `sph_ifs_combine_data`; spherical `database/metadata.py`). Its
sense is on [Conventions](../reference/conventions.md). The main difference in the
workflow is the last step. vlt-sphere hands you the combined cubes, while spherical
goes on to run {term}`TRAP` and measure the companions.

## From the SPHERE Data Center

The [SPHERE Data Center](https://sphere.osug.fr)
([Delorme et al. 2017](https://arxiv.org/abs/1712.06948)) reduces SPHERE data on its
own servers and provides reduced and post-processed data to observers and the public.
spherical instead runs on your machine, starting from the raw frames in the ESO
archive, so you choose every setting and can rerun any step.

## From VIP or pyKLIP

To run your own post-processing on spherical's data, take these files from
`converted/` ([Output products](products.md)).

Cube
: `coro_cube.fits`, shape (wavelength, frame, y, x), the same order as VIP's 4-D cubes
  (`[channels, frames, y, x]`). Use `coro_cube_aligned.fits` from the `align_frames`
  step if your package expects the star on the central pixel.

Angles
: The `DEROT ANGLE` column of `frames_info_coro.csv`, one value per frame. It already
  includes the offsets that a parallactic angle lacks. Check the rotation sense on
  [Conventions](../reference/conventions.md) against what your package expects before
  you pass it on.

Star centre
: Pixel `N // 2` in both axes of an aligned cube. For other cubes,
  `image_centers_fitted_robust.fits` gives `(x, y)` per wavelength and frame.

PSF
: `psf_cube_for_postprocessing.fits`, one unsaturated stellar PSF per wavelength and
  FLUX block, scaled to the exposure time of the CENTER frames with the ND filter
  divided out. Check `DIT_CENTER` against `DIT_CORO` in the database before you use
  it for CORO photometry.

Wavelengths
: `wavelengths.fits`, in nm.

## From another instrument

If you know Keck/NIRC2, SCExAO/CHARIS, GPI or VLT/ERIS, the methods are the same. These
are the SPHERE habits that differ.

- **Two instruments at once.** IFS and IRDIS can record the same star at the same time
  ({term}`IRDIFS`), so one observation gives two database rows with one
  {term}`OBS_ID`.
- **Centring with waffle spots.** The star position comes from {term}`CENTER frame`s
  taken before and after the science frames, or from waffle spots left on throughout
  ({term}`Continuous waffle`), not from the science frames themselves.
- **Flux from separate frames.** Photometry is calibrated on {term}`FLUX frame`s taken
  off the coronagraph through an {term}`ND filter`, normally at the start and end of
  the sequence.
- **One angle per frame.** spherical stores `DEROT ANGLE`, which includes the pupil and
  instrument offsets and true north, instead of the raw parallactic angle.
- **charis for the IFS.** The IFS cubes are extracted with charis, the pipeline
  written for CHARIS and adapted to SPHERE
  ([Samland et al. 2022](https://doi.org/10.1051/0004-6361/202244587)).
- **Anamorphism.** SPHERE images are slightly stretched along one axis
  ({term}`Anamorphism`). spherical leaves the data as they are and TRAP corrects it.

## Terms

| You may know it as | In spherical |
|---|---|
| satellite spots | {term}`Waffle spots` |
| unsaturated PSF, off-axis PSF | FLUX frames, `psf_cube_for_postprocessing.fits` |
| science frames, coronagraphic frames | CORO frames, `coro_cube.fits` |
| derotation angles, `*_derot.fits` | `DEROT ANGLE` |
| observation, epoch | {term}`Observation sequence`, one row of the observation table |

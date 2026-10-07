# Glossary

Look up a SPHERE or spherical term used in these pages.

```{glossary}
ADI
  Angular differential imaging. The derotator keeps the telescope pupil fixed
  on the detector, so the sky rotates during the sequence while the star's
  speckles stay nearly fixed. This separates a companion from the speckles.

Anamorphism
  SPHERE's common-path optics stretch the image by about 0.6 % along one
  detector axis. spherical leaves the data uncorrected and {term}`TRAP` corrects
  it in its {term}`Forward model` (`yx_anamorphism`). See
  [Conventions](conventions.md).

CENTER frame
  An exposure with the {term}`Waffle spots` switched on, from which spherical
  measures where the star sits behind the coronagraph.

charis
  The CHARIS data reduction pipeline, adapted for the SPHERE {term}`IFS`
  ([Samland et al. 2022](https://doi.org/10.1051/0004-6361/202244587)). spherical
  uses it to extract spectral cubes.

Continuous waffle
  A sequence whose science frames are {term}`CENTER frame`s, so the
  {term}`Waffle spots` stay on throughout. The `WAFFLE_MODE` column is set when the
  CENTER frames hold more exposure time than the {term}`CORO frame`s
  (`observation_table.select_primary_science_frames`).

Contrast
  The flux of a companion divided by the flux of its host star in the same band,
  as {term}`TRAP` reports it.

CORO frame
  A science exposure with the star behind the coronagraph.

Coronagraph
  Optics that block most of the starlight so that faint sources close to the star
  can be seen. SPHERE's coronagraphs are described by
  [Beuzit et al. 2019](https://doi.org/10.1051/0004-6361/201935251).

DBI
  Dual-band imaging with {term}`IRDIS`. Two images are taken at once through
  neighbouring filters, for example `DB_K12`. It is the only IRDIS mode the
  pipeline is validated for.

DEROT ANGLE
  The angle spherical computes for every frame from the {term}`Parallactic angle`,
  the pupil offset, the instrument offset and {term}`True north`
  (`database/metadata.py`, `compute_angles`). {term}`TRAP` uses it to place the
  companion model in each frame. See [Conventions](conventions.md) for its sense.

Detection map
  {term}`TRAP`'s map of the measured {term}`Contrast`, its uncertainty and their
  ratio, the signal-to-noise, at every position it tested. See
  [Output products](../concepts/products.md).

DPI
  Dual-polarisation imaging with {term}`IRDIS`. The database covers it for
  discovery and download. spherical does not reduce it.

Field rotation
  How far the {term}`DEROT ANGLE` changes between the first and the last frame of
  a sequence (`ROTATION` column). More rotation lets {term}`ADI` separate a
  companion from the speckles closer to the star.

FLUX frame
  A short exposure with the star moved off the coronagraph, used to calibrate the
  companion's flux.

Forward model
  {term}`TRAP`'s model of how a companion at a given position appears in each
  frame. TRAP fits it together with the systematics instead of subtracting a
  reference image.

HCI_READY
  Database column that marks sequences the reduction can use. It requires
  {term}`CENTER frame`s and {term}`FLUX frame`s, one exposure time (DIT) across
  the CENTER frames and one across the {term}`CORO frame`s, and derotator angles
  that could be computed (`observation_table.compute_hci_ready`). It does not
  require {term}`Pupil tracking`. `usable_only=True` adds pupil tracking and a
  minimum total science exposure of 5 minutes (`sphere_database.usable_mask`).
  Vetting sets it to false when `VETTING_FLAG` is set
  (`observation_table.create_observation_table`).

IFS
  SPHERE's integral field spectrograph
  ([Claudi et al. 2008](https://doi.org/10.1117/12.788366)). It records 39
  wavelength channels from Y to J (`OBS_YJ`) or from Y to H (`OBS_H`), at
  7.46 mas per pixel.

Inverse variance
  The per-pixel weight 1 / variance. spherical writes inverse-variance cubes next
  to the data cubes and can pass them to {term}`TRAP`.

IRDIFS
  Observing mode in which {term}`IRDIS` and {term}`IFS` record the same sequence
  at the same time. Both rows in the database share one {term}`OBS_ID`.

IRDIS
  SPHERE's dual-band imager and polarimeter
  ([Dohlen et al. 2008](https://doi.org/10.1117/12.789786)), at 12.25 mas per
  pixel.

Leaf step
  A pipeline step whose output nothing later reads (`StepSpec.leaf`). Forcing it
  re-runs only itself. Currently this is `align_frames`.

ND filter
  Neutral-density filter that dims the star for the {term}`FLUX frame`s. spherical
  divides out its attenuation (`steps/flux_psf_calibration.py`).

OBS_ID
  ESO's identifier of one observation block. The {term}`IFS` and {term}`IRDIS`
  rows of an {term}`IRDIFS` sequence share it.

Observation sequence
  The frames of one target taken in one observation block, normally
  {term}`FLUX frame`, {term}`CENTER frame`, {term}`CORO frame`s, CENTER, FLUX.
  One row of the observation table.

Parallactic angle
  The angle between the directions to the celestial pole and to the zenith, seen
  from the target. It changes as the target crosses the sky (`PARANG` column).

Programme ID
  ESO programme and run, for example `095.C-0298(D)`. A programme can have several
  runs, lettered (A), (B) and so on (`OBS_PROG_ID` column).

Pupil tracking
  Derotator mode (`ELEV`) that keeps the telescope pupil fixed on the detector, so
  the sky rotates. {term}`ADI` needs it.

Resume
  spherical's default of skipping an enabled step whose declared outputs already
  exist (`step_registry.should_run`). A few steps decide for themselves, and two
  without outputs run every time (see [The configuration model](../concepts/configuration.md)).
  See [Re-run part of a reduction](../how-to/rerun.md).

SAM
  Sparse aperture masking. The database covers it for discovery and download.
  spherical does not reduce it.

SDI
  Spectral differential imaging. Speckles move outwards with wavelength while a
  companion stays in place, which separates them across the {term}`IFS` channels
  or the two {term}`DBI` filters.

Speckles
  Starlight scattered by the wavefront errors the adaptive optics leave, which
  shows up as spots that can look like a companion. They are the noise that
  {term}`ADI`, {term}`SDI` and {term}`TRAP` remove.

TRAP
  Temporal reference analysis of planets
  ([Samland et al. 2021](https://doi.org/10.1051/0004-6361/201937308)), the
  post-processing spherical runs. It models the time series of each pixel with a
  {term}`Forward model` of the companion plus the systematics.

True north
  The correction from the detector's nominal orientation to celestial north,
  -1.75° in spherical (`compute_angles`, `true_north`).

Waffle spots
  Four satellite images of the star, created by a pattern on the deformable
  mirror and visible in {term}`CENTER frame`s. spherical measures the star's
  position from them.

ZIMPOL
  SPHERE's visible-light imaging polarimeter
  ([Schmid et al. 2018](https://doi.org/10.1051/0004-6361/201833620)). spherical
  does not cover it.
```

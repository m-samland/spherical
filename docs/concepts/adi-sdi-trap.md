# ADI, SDI and TRAP

After reading this page you can explain how spherical separates a companion from the
star's speckles, and you know what TRAP's outputs measure.

## The problem

The {term}`Coronagraph` blocks most of the starlight, but not all of it. What is left
forms {term}`Speckles`, spots of starlight with the size of a point source, which can
look like a companion and change slowly during the night. Companions are faint by
comparison. In the 51 Eri example on 2015-09-24, 51 Eri b has a {term}`Contrast` of
about 7 × 10⁻⁶ at 2110 nm, measured by spherical in the IRDIS `DB_K12` data with
`temporal_components_fraction=0.20`. Post-processing exists to model the speckles and
take them out.

## Angular differential imaging

In {term}`Pupil tracking` the speckles stay nearly fixed on the detector while the
sky, and with it the companion, turns around the star. Over a sequence the field
rotates by the angle in the `ROTATION` column. {term}`ADI` uses this difference.

The classic way builds a model of the speckles from the frames themselves, for example
their median or their principal components (KLIP), subtracts it from each frame,
rotates each frame north up and combines them. This is called derotate and stack. Close
to the star a companion moves only a little during the sequence, so part of it ends
up in the speckle model and is subtracted with it. This self-subtraction limits how
close to the star these methods work.

## Spectral differential imaging

Speckles are diffraction features, so they move outwards with wavelength, while a
companion stays at the same position in every channel. {term}`SDI` uses this
difference across the 39 {term}`IFS` channels or between the two filters of an
{term}`IRDIS` {term}`DBI` image. The companion's spectrum then also helps to tell it
apart from the star's.

## TRAP

{term}`TRAP` ([Samland et al. 2021](https://doi.org/10.1051/0004-6361/201937308)) works
with the time series of single pixels instead of whole images. For every position it
tests, it takes the light curve of each pixel the companion would cross. Because the
field rotates, the companion passes through such a pixel only for part of the
sequence, and its expected signal there is known from the angles. TRAP models the
light curve as

- a combination of the main temporal patterns, the principal components, of other
  pixels that the companion does not touch, which describes the speckles and other
  systematics, and
- a {term}`Forward model` of the companion's signal passing through the pixel.

Both are fitted together. The fitted companion amplitude is its contrast, the fit also
gives its uncertainty, and their ratio is the signal-to-noise. The share of principal
components used is `temporal_components_fraction` ([TRAP settings](../reference/trap-settings.md)).

Compared with derotate and stack, the frames are never rotated or interpolated, and
the companion is part of the model instead of being subtracted with the speckles. This
avoids self-subtraction and is why TRAP reaches deeper close to the star (Samland
et al. 2021). TRAP needs the {term}`DEROT ANGLE` of every frame to place the forward
model, and spherical provides it ([Conventions](../reference/conventions.md)).

## From maps to detections

TRAP repeats the fit for every position in the search region and writes a
{term}`Detection map` for every reduced wavelength channel. It then normalises the
signal-to-noise by its spread at each separation, so that a value of 5 means the same
at every radius. Peaks above `candidate_threshold` become candidates, and candidates
above `detection_threshold` count as detections ([TRAP settings](../reference/trap-settings.md)).

For a detection TRAP measures the position in every channel and the contrast spectrum.
It also combines the channels using three template spectra, which raises the
signal-to-noise for companions that resemble one of them.

`flat`
: The same contrast in every channel.

`L-type`
: A 1500 K atmosphere model (DRIFT-PHOENIX).

`T-type`
: A 760 K atmosphere model with clouds (petitCODE).

The templates are defined in TRAP's `detection.py`. The files they produce are listed
on [Output products](products.md).

Next: [The observation database](database.md)

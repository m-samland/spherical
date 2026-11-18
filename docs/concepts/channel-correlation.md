---
stage: trap
---

# The channel correlation of a spectrum

After reading this page you can build the covariance matrix of a spectrum from
spherical's outputs and use it in a fit, and you know how {term}`TRAP` measures and models the
correlation of the noise between wavelength channels.

## Why the channels are correlated

{term}`Speckles` are diffraction features, so a speckle covers several neighbouring
{term}`IFS` channels before it moves outwards with wavelength. Where the companion
sits on a speckle in one channel, it sits on the same speckle in the next channel.
The noise of neighbouring channels is therefore correlated.

A fit that treats the channels as independent counts each correlated channel as new
information. It then gives too small uncertainties of the fitted parameters and can
prefer the wrong model
([Greco and Brandt 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...833..134G/abstract)).

## How TRAP measures the correlation

The correlation coefficient {math}`\Psi_{ij}` of channels {math}`i` and {math}`j` is
the covariance of their noise divided by the product of their standard deviations. It
is 1 for a channel with itself and 0 for independent channels.

TRAP measures it on the maps of the {term}`Contrast`, in rings of 9 pixels around the star, about two
{term}`λ/D` for IFS. For every pair of channels it correlates the values of all pixels
of a ring that are finite in every channel and outside the noise mask
([The noise mask](products.md#the-noise-mask)). The mask keeps a bright companion out
of its own noise estimate.

A ring holds far fewer independent speckles than there are pairs of channels. The
measured matrix is therefore noisy and often has no inverse, which a fit needs.

## The model

TRAP fits each ring with the model of Greco and Brandt (2016), extended by a term common
to all channels,

```{math}
\Psi_{ij} = A_\rho \exp\left[-\frac{1}{2}\left(\frac{\rho}{\sigma_\rho}\,\frac{\lambda_i-\lambda_j}{\lambda_c}\right)^2\right]
          + A_\lambda \exp\left[-\frac{1}{2}\left(\frac{\lambda_i-\lambda_j}{\sigma_\lambda\,\lambda_c}\right)^2\right]
          + A_c + A_\delta\,\delta_{ij}.
```

Here {math}`\lambda_c` is the mean wavelength of the channels, {math}`\rho` the
separation in units of {math}`\lambda_c/D`, and {math}`\delta_{ij}` is 1 for
{math}`i = j` and 0 otherwise.

The first term describes the speckles. Between two channels a speckle at separation
{math}`\rho` moves by {math}`\rho\,(\lambda_i-\lambda_j)/\lambda_c`, so the farther
it is from the star, the faster its correlation falls off with wavelength. The length
{math}`\sigma_\rho` is the distance a speckle has to move before the correlation is
lost.

The second term is correlated over the same fraction {math}`\sigma_\lambda` of the
wavelength at every separation, for example from the extraction of the IFS spectra.
The third term, {math}`A_c`, is noise common to all channels, such as a change of the
residual starlight that affects every wavelength in the same way. Greco and Brandt do
not have this term. Without it one of the two kernels widens to take its place, and the
lengths then no longer describe the speckles or the spectra. The last term is noise that
is not correlated between channels. The diagonal of {math}`\Psi` is 1, so
{math}`A_\delta = 1 - A_\rho - A_\lambda - A_c`.

The amplitudes are fitted per ring, because the mix of speckles and other noise changes
with separation. {math}`\sigma_\lambda` is fitted once per reduction and shared by all
rings, because it describes the instrument.

{math}`\sigma_\rho` is held at 1 λ/D by default (`spectral_correlation_sigma_rho`). The
rings of a typical reduction cover too small a range of separation to tell
{math}`\sigma_\rho` apart from {math}`\sigma_\lambda`. When they cover a wide range,
set the setting to None to fit it.

{math}`A_\delta` is held at 0.01 or more. The smallest eigenvalue of the matrix is then
about {math}`A_\delta` or larger, so the matrix always has an inverse.

## Checking the fit

The `FIT` table of `spectral_correlation_{run}.fits` gives the fitted parameters of
every ring and the root-mean-square (rms) difference between the model and the
measured matrix. A ring whose `rms` exceeds `spectral_correlation_flag_rms` (0.2) is
`flagged` and not used for the matrices between rings. Many flagged rings, or an
`rms` that grows with separation, mean the model does not describe your data well.

## The matrix at a companion's separation

`trap.spectral_correlation.correlation_at` gives the matrix at any separation in
pixels. It uses the shared lengths and interpolates the amplitudes linearly between
the neighbouring rings that are not flagged. Inside the innermost and beyond the
outermost ring it keeps the amplitudes of that ring.

With fewer than four channels, as for {term}`IRDIS`, TRAP does not fit the model, and
`correlation_at` returns the measured matrix of the nearest ring.

## The covariance of a spectrum

For a spectrum {math}`c_i` with uncertainties {math}`\sigma_i` the covariance matrix is

```{math}
\Sigma_{ij} = \sigma_i\,\sigma_j\,\Psi_{ij}.
```

You compare a model spectrum {math}`m_i` with the data through

```{math}
\chi^2 = (c - m)^\mathsf{T}\,\Sigma^{-1}\,(c - m),
```

which reduces to the usual sum of squared, normalised residuals when {math}`\Psi` is
the identity.

The error of the star's flux scales the whole spectrum of a night
([What the uncertainties include](spectra.md#what-the-uncertainties-include)). For a
relative error {math}`s` add {math}`s^2\,m_i\,m_j` to {math}`\Sigma_{ij}`, or fit a
free scale factor for each night instead.

This example builds the covariance of a companion's spectrum from an IFS reduction.
Replace `reduction_directory` with your `config.directories.reduction_directory`. The
`*` in `*_51_Eri` is part of the SIMBAD name, not a wildcard.

```python
from pathlib import Path

import numpy as np
import pandas as pd
from trap.spectral_correlation import correlation_at, read_stack

reduction_directory = Path("/data/sphere/reduction")
folder = reduction_directory / "IFS/trap/*_51_Eri/OBS_H/2015-09-24"
(stack_file,) = folder.glob("spectral_correlation_*_frac0.15_temporal.fits")
psi, fit, wavelengths = read_stack(stack_file)

# 51 Eri b is validated with the T-type template.
table = pd.read_csv(folder / "template_matching/validated_companion_table_short_T-type.csv")
spectrum = table[(table["candidate_id"] == 0) & table["contrast"].notna()]
# The stack holds only the reduced channels. Match them by wavelength, both in µm.
channels = [int(np.argmin(np.abs(wavelengths - w))) for w in spectrum["wavelength"]]
separation = float(spectrum["separation"].iloc[0])  # pixels

correlation = correlation_at(separation, psi, fit, wavelengths)[np.ix_(channels, channels)]
sigma = spectrum["uncertainty"].to_numpy()
covariance = sigma[:, None] * sigma[None, :] * correlation
print(covariance.shape)
print(fit[["separation", "a_rho", "a_lambda", "a_const", "a_delta", "sigma_lambda", "rms"]].round(3).head(3))
print(np.round(correlation[:4, :4], 2))
```

```{expected-result}
For 51 Eri b on 2015-09-24 the covariance has the shape (37, 37), one row per reduced
IFS channel. The first rings of the fit and the correlation of the first four channels
at the companion's separation of 60.7 pixels are

    (37, 37)
       separation  a_rho  a_lambda  a_const  a_delta  sigma_lambda    rms
    0        48.0  0.722     0.223      0.0    0.054         0.026  0.197
    1        51.0  0.748     0.186      0.0    0.065         0.026  0.170
    2        54.0  0.776     0.148      0.0    0.076         0.026  0.150
    [[1.   0.87 0.78 0.67]
     [0.87 1.   0.87 0.78]
     [0.78 0.87 1.   0.87]
     [0.67 0.78 0.87 1.  ]]

Neighbouring channels are correlated at 0.87, and the correlation falls off over a few
channels. A `ValueError` from the unpacking of `folder.glob` means the run has no
`spectral_correlation_*.fits`, see the note below.
```

```{common-mistake}
Match the channels by wavelength, not by `wavelength_index`. The stack holds only the
reduced channels, 37 of the 39 IFS channels, while `wavelength_index` counts all 39.
```

```{note}
The noise mask and the channel correlation come from the TRAP detection step. A
reduction made with a TRAP version that did not write them gets them when you run the
detection step again ([Re-run part of a reduction](../how-to/rerun.md)).
```

Next: [Coming from another pipeline or instrument](other-pipelines.md)

---
stage: trap
---

# Spectra and photometry

After reading this page you know what the contrast spectra and photometry of spherical
measure, which position they were measured at, and what their uncertainties include.
The files are listed on [Output products](products.md).

## What a spectrum measures

{term}`TRAP` fits the companion's signal in every wavelength channel on its own
([ADI, SDI and TRAP](adi-sdi-trap.md)). The result for one channel is the
{term}`Contrast`, the companion's flux divided by the star's flux in that channel.

The star's flux comes from the {term}`PSF` measured in the {term}`FLUX frame`s,
corrected for the {term}`ND filter`. The star amplitude scales it to every science
frame, so that changes of the observing conditions during the sequence do not change
the contrast ([The star amplitude](products.md#the-star-amplitude)). With
`apply_coronagraph_transmission` the contrast is also corrected for the transmission
of the {term}`Coronagraph` near the mask ([TRAP settings](../reference/trap-settings.md)).

A contrast spectrum is therefore free of the star's spectrum and of the throughput of
the instrument and the atmosphere. To get the companion's flux, multiply the contrast
by a spectrum of the star integrated over each channel.

For {term}`IFS` this gives one value per wavelength channel. For {term}`IRDIS` it gives
photometry in the two filters of the observation. In a broadband filter both channels
see the same filter.

## Detected and forced spectra

The spectrum of a detected companion is in
`template_matching/validated_companion_table_short_{template}.csv`, measured at the
position TRAP fitted. When the companion is clearly detected, this is the better
position.

The optional step `run_trap_forced_extraction` measures the spectrum at positions you
give, for example the position an orbit predicts, and writes `forced_spectra_{run}.csv`.
At low signal-to-noise a fitted position is drawn towards the brightest noise peak
nearby, which raises the measured contrast. A predicted position avoids this, as long
as the prediction is accurate to a fraction of the {term}`FWHM`.

This forced photometry also gives a result where nothing is detected. Its values then
scatter around zero within their uncertainties, and negative values are part of that
scatter.

```{common-mistake}
Do not drop negative or insignificant forced values before you average nights or fit a
model. Only the upward fluctuations would remain, and the result would be biased upwards.
Keep every value with its uncertainty.
```

## What the uncertainties include

The uncertainty of each channel is TRAP's fit uncertainty multiplied by
`snr_normalization`, the spread of the signal-to-noise at the companion's separation.
TRAP measures that spread over all pixels at the separation, with sources left out
([The noise mask](products.md#the-noise-mask)). For the detected spectra these are the
configured companions and the validated detections. For forced spectra they are the
companions configured for the target. The
uncertainty is therefore the noise measured in your data at that separation, from
{term}`Speckles`, photon noise and detector noise together.

The uncertainty does not include the error of the star's flux. That error is common to
all channels of a night and scales the whole spectrum by the same factor.
`SCALEUNC` in the header of `star_amplitude.fits` gives an estimate of it from the
disagreement of the FLUX blocks. When you compare nights or combine them, give each
night its own scale factor or add the term to the covariance
([The channel correlation of a spectrum](channel-correlation.md)).

The uncertainties of neighbouring channels are also not independent. Fit a spectrum
with the covariance described on that page, not with the uncertainties alone.

Next: [The channel correlation of a spectrum](channel-correlation.md)

# SPHERE in five minutes

After reading this page you know enough about SPHERE's instruments, observing modes and
frame types to follow the rest of these pages.

## The instrument

SPHERE is the high-contrast imager on the third unit telescope (UT3) of ESO's Very
Large Telescope. An extreme adaptive optics system, SAXO, corrects the turbulence, a
{term}`Coronagraph` blocks most of the starlight, and three science instruments
record what is left ([Beuzit et al. 2019](https://doi.org/10.1051/0004-6361/201935251)).

{term}`IRDIS`
: A dual-band imager that records two images of the same field side by side on one
  detector, through two neighbouring filters. It covers 0.95 to 2.4 µm over an
  11″ × 11″ field at 12.25 mas per pixel
  ([Dohlen et al. 2008](https://doi.org/10.1117/12.789786); Beuzit et al. 2019,
  section 8). Besides {term}`DBI` it has classical imaging with one broad-band
  filter, polarimetry ({term}`DPI`) and long-slit spectroscopy.

{term}`IFS`
: An integral field spectrograph. A lenslet array cuts the central field of about
  1.73″ × 1.73″ into some 23 000 spectra, which spherical turns into cubes of 39
  wavelength channels. It runs in two modes, Y to J (`OBS_YJ`, up to about 1.35 µm,
  spectral resolution about 50) and Y to H (`OBS_H`, up to about 1.65 µm)
  ([Claudi et al. 2008](https://doi.org/10.1117/12.788366); Beuzit et al. 2019,
  section 7).

{term}`ZIMPOL`
: A visible-light imaging polarimeter for 510 to 900 nm with a field of about
  3.6″ × 3.6″ ([Schmid et al. 2018](https://doi.org/10.1051/0004-6361/201833620)).
  spherical does not cover it.

## IRDIS and IFS together

IRDIS and IFS can observe the same star at the same time, each with its own detector
and exposures. In the {term}`IRDIFS` mode IFS runs in Y to J and IRDIS most often in the
`DB_H23` filter pair. In IRDIFS_EXT, IFS runs in Y to H and IRDIS in `DB_K12`, which covers
the Y, J, H and K bands in one observation (Beuzit et al. 2019, sections 7 and 8). The
database has one row for each instrument, and the two rows share the
{term}`OBS_ID` of the observation block.

## Frame types

An observation block mixes exposures of different types, set in the header keyword
`DPR TYPE` (Beuzit et al. 2019, section 9).

{term}`CORO frame` (`OBJECT`)
: The science frames, with the star behind the coronagraph.

{term}`CENTER frame` (`OBJECT,CENTER`)
: A periodic pattern on the deformable mirror creates four satellite images of the
  star outside the coronagraph, the {term}`Waffle spots`. spherical fits them to find
  where the star sits behind the coronagraph, and follows their brightness to track
  the stellar flux.

{term}`FLUX frame` (`OBJECT,FLUX`)
: The star is moved off the coronagraph and dimmed with an {term}`ND filter`, which
  gives an unsaturated image of the stellar PSF for flux calibration.

A typical sequence runs FLUX, CENTER, CORO, CENTER, FLUX. Some sequences keep the
waffle spots on during the whole science part ({term}`Continuous waffle`). Calibration
frames come from separate blocks. For IFS the pipeline downloads the wavelength
calibration (`WAVECAL`), for IRDIS the flat fields and background frames for the
science exposures, taken from the sky, the instrument background or a dark.

```{instrument-background}
During a sequence the derotator either holds the telescope pupil fixed on the
detector ({term}`Pupil tracking`), so that the sky rotates, or holds the sky fixed.
The database records this in `DEROTATOR_MODE`. Most high-contrast sequences use pupil
tracking, because the rotation is what {term}`ADI` needs. [Conventions](../reference/conventions.md)
gives the angle spherical computes for each frame.
```

## Sources

- J.-L. Beuzit et al. 2019, SPHERE: the exoplanet imager for the Very Large
  Telescope, A&A 631, A155, [doi:10.1051/0004-6361/201935251](https://doi.org/10.1051/0004-6361/201935251)
- K. Dohlen et al. 2008, The infra-red dual imaging and spectrograph for SPHERE: design and performance,
  Proc. SPIE 7014, 70143L, [doi:10.1117/12.789786](https://doi.org/10.1117/12.789786)
- R. U. Claudi et al. 2008, SPHERE IFS: the spectro differential imager of the VLT
  for exoplanets search, Proc. SPIE 7014, 70143E, [doi:10.1117/12.788366](https://doi.org/10.1117/12.788366)
- H. M. Schmid et al. 2018, SPHERE/ZIMPOL high resolution polarimetric imager, A&A
  619, A9, [doi:10.1051/0004-6361/201833620](https://doi.org/10.1051/0004-6361/201833620)
- [ESO SPHERE User Manual](https://www.eso.org/sci/facilities/paranal/instruments/sphere/doc.html)

Next: [ADI, SDI and TRAP](adi-sdi-trap.md)

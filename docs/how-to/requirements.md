# Practical requirements

Use this page to estimate the disk space and run time of a reduction before you start
it. The numbers come from the two 51 Eri tutorials and are rough, since they were measured on
a shared server where other jobs change the run time.

## Measured on 51 Eri

Both tutorials reduce the same sequence of 2015-09-24, 68 minutes of science exposure
with 256 {term}`CORO frame`s. They ran on an AMD EPYC 9454 node with `set_ncpu(48)`.
The line at the top of each tutorial records the versions and the date.

| | IRDIS `DB_K12` | IFS `OBS_H` |
|---|---|---|
| Frames (CORO, CENTER, FLUX) | 256, 8, 60 | 256, 4, 30 |
| Raw science files | 2.7 GB | 4.9 GB |
| Pre-processing | 2.6 min | 9.5 min |
| TRAP on the tutorial annulus | 2.5 min (31 to 43 px) | 37 min (50 to 72 px) |
| TRAP on the template's whole search region, earlier runs | about 90 min (1 to 200 px) | about 65 min (1 to 65 px) |
| Reduced cubes (`converted/`) | 6.5 GB | 6.9 GB |
| Extracted cubes of single frames | none | 13 GB |
| Calibrations | 0.02 GB | 2.2 GB |
| TRAP results | 0.01 GB | 0.1 GB |

The times of every step are in
{download}`requirements_51eri_steps.csv <../_data/requirements_51eri_steps.csv>` and the
disk use per stage in
{download}`requirements_51eri_disk.csv <../_data/requirements_51eri_disk.csv>`. The
whole-region times come from earlier runs of the same sequence on the same node and are
not in these files. The download is not included either, since it depends on your
connection and the ESO archive.

## Run time

Pre-processing is short. For IRDIS most of it is the flux PSF calibration, whose time
does not change with the number of cores, and the whole IRDIS pre-processing took 3 to
4 minutes with 4 cores as with 48. For IFS the extraction of the spectral cubes with
charis takes most of the time. On a busy node it took 10 minutes with 48 cores and
11.5 minutes with 24.

TRAP takes most of the time. Its cost grows with the number of frames, the number of
channels it uses (37 for IFS, 2 for IRDIS) and the area of the search region, and the
cost per position grows with the separation from the star. An annulus around a known
companion is therefore much faster than a search of the whole field, as the table shows
for IRDIS. For IFS the 50 to 72 px annulus reaches the outer edge of the field, so it
saves little. Run on the same day under the same load, it took 63 minutes and the whole
search region 64. The 37 minutes in the table come from a quieter day.

## Disk space

The raw files of a sequence are larger than the `TOTAL_FILE_SIZE_MB` column of the
observation table, which counts only the science frames (2.0 GB for this IRDIS row and
4.1 GB for the IFS row). The FLUX and CENTER frames and the calibrations come on top.

The reduced cubes hold every CORO frame with its {term}`Inverse variance`, so their size
is the number of frames times the number of channels times the image size times 4 bytes,
twice. For IRDIS that is 256 × 2 × 1024² × 4 bytes, or 2.1 GB, for the science cube and
the same for its inverse variance. An IFS reduction also keeps the extracted cube of every single frame
and the charis calibration until you remove them.

[Output products](../concepts/products.md) explains what `cleanup_pipeline_products`
deletes after a finished IFS reduction. A planned estimate of disk space and time before
a run is tracked in [#222](https://github.com/m-samland/spherical/issues/222).

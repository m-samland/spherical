# Anatomy of a reduction

After reading this page you know which steps a reduction runs, which package does the
work in each, and how IFS and IRDIS reductions differ.

## One entry point

`execute_targets(observations, config)` reduces a list of observations, one after the
other. For each it reads the instrument from the table row and hands the observation
to the IRDIS reduction (`irdis_reduction.execute_irdis_target`) or to the IFS
reduction (`ifs_reduction.execute_target`). The configuration must match the
instrument. An `IFSReductionConfig` with an IRDIS observation, or the reverse, stops
with a `ValueError` (`ifs_reduction.py`, `execute_targets`). With `config=None` each
observation gets its instrument's default configuration.

## The steps

```{step-diagram}
```

Each name is a switch in `config.steps`, and [Pipeline steps](../reference/steps.md)
gives one sentence per step.

Get the data
: `download_data` fetches the raw science and calibration frames of the sequence from
  the ESO archive, skipping files that are already on disk.

Calibrate and build cubes
: The two instruments differ only here. IFS builds a wavelength calibration and
  extracts a spectral cube from every raw frame with {term}`charis`, then bundles them
  into one cube per frame type. IRDIS builds a master background, flat field and
  bad-pixel map and calibrates the raw frames into cubes directly.

Describe the frames
: `compute_frames_info` works out the time, {term}`Parallactic angle` and
  {term}`DEROT ANGLE` of every frame and writes them to `frames_info_*.csv`.
  `cube_header_update` writes version and provenance keywords into the cube headers.

Find the star
: `find_centers` fits the {term}`Waffle spots` in the {term}`CENTER frame`s to locate
  the star in every frame and wavelength. `process_extracted_centers` cleans up those
  fits. IFS fits a smooth curve across wavelength in each frame, IRDIS flags outliers
  over time and fills in failed fits. `plot_image_center_evolution` plots the result.

Calibrate the flux
: `calibrate_spot_photometry` measures the waffle spots, `calibrate_flux_psf` builds
  the unsaturated stellar PSF from the {term}`FLUX frame`s, and `spot_to_flux` ties the
  two together to follow the stellar flux through the sequence.

Optional products
: `align_frames` writes a copy of the science cube with the star on the central pixel,
  for your own ADI or PCA. It is off by default and nothing later reads it.

Find companions
: `run_trap_reduction` and `run_trap_detection` run {term}`TRAP` (see
  [Where TRAP acts](#where-trap-acts)).

## Where charis acts

For IFS, `reduce_calibration` and `extract_cubes` call charis, the CHARIS data
reduction pipeline adapted for SPHERE
([Samland et al. 2022](https://doi.org/10.1051/0004-6361/202244587); code at
[m-samland/charis-dep](https://github.com/m-samland/charis-dep)). charis builds the
wavelength solution from the `WAVECAL` frames and extracts the spectrum of every
lenslet from each raw frame. The steps from `bundle_output` on are spherical's own
code.

IRDIS needs no charis. Its calibration and pre-processing draw partly on A. Vigan's
SPHERE tools ([Vigan 2020](https://ascl.net/2009.002);
[avigan/SPHERE](https://github.com/avigan/SPHERE)), on which the astrometric and
photometric calibration of both instruments also builds.

(where-trap-acts)=
## Where TRAP acts

The two TRAP steps do not run inside `execute_targets`. They run when your script calls
`run_trap_on_observations(observations, trap_config, reduction_config, species_database_directory)`
after `execute_targets`, as both templates do. That function reads the reduced cubes,
the angles in `frames_info_*.csv`, the star positions and the stellar PSF, and writes
its results into a separate TRAP folder ([Output products](products.md)). The steps
`run_trap_reduction` and `run_trap_detection` in `config.steps` switch these two parts
on and off. [ADI, SDI and TRAP](adi-sdi-trap.md) explains what TRAP does.

## IFS and IRDIS side by side

| | IFS | IRDIS |
|---|---|---|
| Raw frames to cubes | charis extraction, 39 channels | background, flat and bad-pixel correction, 2 channels |
| Science cube for 51 Eri, 2015-09-24 | 39 × 256 × 262 × 262 | 2 × 256 × 1024 × 1024 |
| Cleaning the star positions | smooth fit across wavelength per frame | outliers flagged over time |
| Products directory | `{method}/converted/`, for example `optext/converted/` | `converted/` |
| Angle file per frame type | `*_parallactic_angles.fits` (holds `DEROT ANGLE`) | none, angles only in `frames_info_*.csv` |
| Bad pixels for TRAP | from the {term}`Inverse variance` | `badpixel_map.fits`, written by `preprocess_irdis` |

Shapes are (wavelength, frame, y, x). [Conventions](../reference/conventions.md) has the
axis order, units and centre of every file.

## When something fails

Each observation is reduced on its own. If a step raises an error, the reduction of
that observation stops, the error is logged and written to `crash_report.txt` in the
observation's directory, and `execute_targets` goes on with the next observation.
`run_trap_on_observations` behaves the same way and writes `trap_crash_report.txt` into
the TRAP folder, removing an old report when it starts a new attempt
(`run_trap.py`). [Monitor runs](../how-to/monitor.md) shows how to collect these
reports across many targets.

## Running again

When you run a script a second time, each enabled step whose outputs already exist is
skipped ({term}`Resume`), so a finished observation costs almost nothing.
[The configuration model](configuration.md) explains the rules and
[Re-run part of a reduction](../how-to/rerun.md) shows how to recompute chosen steps.

Next: [The configuration model](configuration.md)

# The configuration model

After reading this page you can change any setting of a reduction and know when the
change takes effect.

## Configuration objects

A reduction takes one configuration object, `IFSReductionConfig` for IFS and
`IRDISReductionConfig` for IRDIS (`pipeline/pipeline_config.py`). Each is a dataclass
made of smaller dataclasses, one per concern, plus a few switches of its own that
control what spherical hands to TRAP.

::::{tab-set}
:sync-group: instrument

:::{tab-item} IFS
:sync: ifs

| Part | Holds |
|---|---|
| `config.calibration` | the charis wavelength calibration |
| `config.extraction` | the charis cube extraction |
| `config.preprocessing` | flux calibration, frame selection and ESO download settings |
| `config.directories` | where raw data and products go |
| `config.resources` | CPUs per stage |
| `config.steps` | which steps run, and `force` |
| `config.alignment` | the optional `align_frames` step |
:::

:::{tab-item} IRDIS
:sync: irdis

| Part | Holds |
|---|---|
| `config.calibration` | the master background, flat and bad-pixel map |
| `config.irdis_preprocessing` | cropping, bad-pixel correction, anamorphism and detector noise for the raw frames |
| `config.preprocessing` | flux calibration, frame selection and ESO download settings |
| `config.directories` | where raw data and products go |
| `config.resources` | CPUs per stage |
| `config.steps` | which steps run, and `force` |
| `config.alignment` | the optional `align_frames` step |
:::
::::

Every field, with its default and meaning, is in the
[Configuration reference](../reference/configuration.md).

## Changing a setting

You can assign a field directly,

```python
from pathlib import Path

from spherical.pipeline.pipeline_config import IFSReductionConfig

config = IFSReductionConfig()
config.directories.base_path = "/data/sphere"
print(config.directories.base_path)
print(config.directories.reduction_directory == Path.home() / "data/sphere/reduction")
```

```text
/data/sphere
True
```

or replace a whole part with a modified copy. `merge` returns a copy with the named
fields changed and leaves the original as it was.

```python
from spherical.pipeline.pipeline_config import IFSReductionConfig

config = IFSReductionConfig()
config.steps.merge(align_frames=True)
print(config.steps.align_frames)
config.steps = config.steps.merge(align_frames=True)
print(config.steps.align_frames)
```

```text
False
True
```

```{common-mistake}
`config.steps.merge(...)` on its own line changes nothing, as the first `print`
shows. Assign the result back, `config.steps = config.steps.merge(...)`.
```

```{common-mistake}
`raw_directory` and `reduction_directory` are derived from `base_path` only when the
configuration is created. Changing `base_path` afterwards leaves them where they were,
as the first example shows. Set all three, as the reduction templates do.
```

Directory fields are made absolute as soon as you set them, with `~` expanded
(`DirectoryConfig`). A relative path therefore refers to the directory you started
the script from.

## The TRAP configuration

TRAP has its own configuration object, made by `trap_config_for_ifs()` or
`trap_config_for_irdis()`. Its parts are frozen, so `merge` is the only way to change
them, for example
`trap_config.reduction = trap_config.reduction.merge(search_region_outer_bound=65)`.
[TRAP settings](../reference/trap-settings.md) lists the settings that matter for SPHERE.

## Steps, resume and force

`config.steps` has one switch per step, named as in the
[step diagram](anatomy.md). By default a reduction resumes ({term}`Resume`). An enabled
step is skipped when all the files it declares as outputs already exist
(`step_registry.should_run`). A few steps follow their own rules.

- `cube_header_update` and `plot_image_center_evolution` declare no outputs and run
  every time.
- `download_data`, `reduce_calibration`, `irdis_calibration` and
  `run_trap_reduction` decide for themselves what is already done. TRAP skips a
  reduction when its detection file exists. That file name contains the number of
  components and `temporal_components_fraction`, so a new fraction runs again while a
  new search region does not.

Resume only checks that files exist. After you change a setting of a finished step,
force it with `config.steps.force`.

`force=True`
: Recompute every enabled step.

`force={"find_centers"}`
: Recompute the named steps and every step after the earliest of them, including the
  TRAP steps when you call `run_trap_on_observations` with the same configuration.

`force={"align_frames"}`
: A {term}`Leaf step` named in `force` re-runs only itself.

A name that is not a step of the instrument stops the run with a `ValueError` that
lists the valid names (`step_registry.validate_force`).
[Re-run part of a reduction](../how-to/rerun.md) has recipes.

## CPUs

`config.resources` sets the number of CPUs per stage with the fields `ncpu_calib`,
`ncpu_extract`, `ncpu_center`, `ncpu_preprocess` and `ncpu_trap`. `config.set_ncpu(n)` sets all five.
When a reduction starts, spherical copies the first three into
`config.calibration.ncpus`, `config.preprocessing.ncpu_cubebuilding` and
`config.preprocessing.ncpu_find_center` (`apply_resources`), so set CPUs on
`config.resources`, not on those fields. `ncpu_preprocess` is read directly by the
IRDIS pre-processing.

TRAP gets its CPU count only from `config.apply_trap_resources(trap_config)`. That call
also resets TRAP's `scratch_dir`, so make it before any change to
`trap_config.reduction`.

```python
from trap.parameters import trap_config_for_ifs

from spherical.pipeline.pipeline_config import IFSReductionConfig

config = IFSReductionConfig()
config.set_ncpu(8)

trap_config = trap_config_for_ifs()
config.apply_trap_resources(trap_config)
trap_config.reduction = trap_config.reduction.merge(scratch_dir="/scratch/trap")

print(config.resources)
print(config.calibration.ncpus, config.preprocessing.ncpu_cubebuilding, config.preprocessing.ncpu_find_center)
print(trap_config.reduction.ncpus, trap_config.reduction.scratch_dir)
```

```text
Resources(ncpu_calib=8, ncpu_extract=8, ncpu_center=8, ncpu_trap=8, ncpu_preprocess=8)
8 8 8
8 /scratch/trap
```

## Keeping a record

There is no configuration file format. Your reduction script is the record of a run,
so keep it with the results. The pipeline writes its version and provenance into the
FITS headers ([Conventions](../reference/conventions.md)).

Next: [Output products](products.md)

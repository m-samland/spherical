# Re-run part of a reduction

After this guide you can recompute chosen steps of a finished reduction without
redoing the rest.

## Why a second run skips work

A reduction resumes by default ({term}`Resume`). An enabled step whose output files
already exist is skipped, and the log says `outputs present` for it. This makes it cheap
to rerun a script, but it also means that a changed setting has no effect on a step
that already finished. To recompute a step, name it in `config.steps.force`.
[The configuration model](../concepts/configuration.md) explains the rules.

## Recipes

Each recipe changes an existing `config` before you call `execute_targets` and
`run_trap_on_observations` as before.

Redo everything
: ```python
  config.steps = config.steps.merge(force=True)
  ```

Redo from the star centring on, for example after changing a centring setting
: ```python
  config.steps = config.steps.merge(force={"find_centers"})
  ```
  This recomputes `find_centers` and every step after it: the centre clean-up and
  plot, the flux calibration, `align_frames` if it is enabled, and both TRAP steps.

Redo only the aligned cube, after changing `config.alignment`
: ```python
  config.steps = config.steps.merge(align_frames=True, force={"align_frames"})
  ```
  `align_frames` is a {term}`Leaf step`, so forcing it recomputes nothing else.

Redo TRAP with new TRAP settings
: ```python
  config.steps = config.steps.merge(force={"run_trap_reduction"})
  ```
  This reruns the TRAP reduction and the detection. You can skip the call to
  `execute_targets` and call only `run_trap_on_observations`, since the reduction
  products do not change.

Redo only the detection, for example with new thresholds
: ```python
  config.steps = config.steps.merge(force={"run_trap_detection"})
  ```
  The TRAP reduction is kept and only the detection and characterisation run again.

The cascades above were checked with `step_registry._forced` and are the same for IFS
and IRDIS. A misspelt name stops the run before anything is computed:

```text
ValueError: Unknown step name(s) in force: ['find_center']. Valid names: ['align_frames', 'bundle_output', 'calibrate_flux_psf', 'calibrate_spot_photometry', 'compute_frames_info', 'cube_header_update', 'download_data', 'extract_cubes', 'find_centers', 'plot_image_center_evolution', 'process_extracted_centers', 'reduce_calibration', 'run_trap_detection', 'run_trap_reduction', 'spot_to_flux']
```

## Things that do not need force

- A new `temporal_components_fraction` runs TRAP again by itself, because TRAP's result
  files are named after it. A new search region or new thresholds do not, so force
  those.
- `cube_header_update` and `plot_image_center_evolution` run every time anyway.

## Deleting files instead

Deleting a step's output files also makes it run again, but only that step. Later
steps keep their old products, which then no longer match. Use `force` instead, which
recomputes everything that depends on the step.

Three steps are not gated by their products but by a hidden marker file:
`.extract_cubes.done`, `.align_frames.done` and `.run_trap_detection.done`
([Output products](../concepts/products.md)). Deleting their products does not rerun
them. Force them, or delete the marker as well.

```{common-mistake}
Do not set TRAP's own `overwrite_reduction` or `overwrite_detection` options to rerun
TRAP. spherical does not read them; it uses `config.steps.force`.
```

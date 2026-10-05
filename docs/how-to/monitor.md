# Monitor runs

After this guide you can see which reductions finished, which failed and why, and look
at all TRAP results at once. The outputs below come from a reduction directory with a
few test reductions.

## Which reductions finished

`reduction_status` reads every reduction log below a directory and prints one line per
observation and pipeline.

```bash
reduction_status ~/data/sphere/reduction --instrument irdis
```

```text
TARGET     INSTR  BAND    NIGHT       PIPELINE   COMPLETE  LAST_STEP                   STATUS
---------------------------------------------------------------------------------------------
*_51_Eri   IRDIS  BB_H    2015-09-25  reduction  False     download_data               started
*_51_Eri   IRDIS  DB_K12  2015-09-24  reduction  True      spot_to_flux_normalization  success
*_51_Eri   IRDIS  DB_K12  2015-09-24  trap       True      trap_session                success
*_51_Eri   IRDIS  DB_K12  2017-09-27  reduction  False     preprocess_irdis            coro_skipped
*_51_Eri   IRDIS  DB_K12  2017-09-27  trap       False     trap_session                failed
*_bet_Pic  IRDIS  DB_K12  2014-12-07  reduction  True      spot_to_flux_normalization  success
*_bet_Pic  IRDIS  DB_K12  2014-12-07  trap       False     trap_reduction              started
*_pi._Men  IRDIS  DB_H23  2015-12-19  reduction  False     download_data               started
*_pi._Men  IRDIS  DB_H23  2015-12-19  trap       True      trap_session                success
```

`PIPELINE`
: `reduction` for `execute_targets`, read from `reduction.jsonlog`; `trap` for
  `run_trap_on_observations`, read from `trap_reduction.jsonlog`.

`COMPLETE`
: Whether the run reached its last required step, `spot_to_flux_normalization` for the
  reduction and the end of the TRAP session for TRAP. It stays true when an optional
  step such as `align_frames` runs or fails afterwards.

`LAST_STEP`, `STATUS`
: The last step the log recorded and how it ended. `started` with `COMPLETE` false
  means the run is still going or was stopped. `failed` means an error, which has a
  crash report.

Add `--pipeline trap` to see only TRAP, and `--csv status.csv` to save the table.
The tool reads only the log of the latest run in each folder. Earlier runs' logs are in
`old_logs/`.

## Why a reduction failed

`crash_reports` collects the crash reports below a directory and counts the errors.

```bash
crash_reports ~/data/sphere/reduction --instrument ifs --top 5
```

```text
DATASET                     INSTR  PIPELINE  EXCEPTION                 MESSAGE
------------------------------------------------------------------------------
*_bet_Pic/OBS_H/2019-03-10  IFS    trap      numpy.linalg.LinAlgError  SVD did not converge

🔢 Most frequent exceptions:
numpy.linalg.LinAlgError: 1
```

The full traceback is in the report itself, `crash_report.txt` in the observation
folder for the reduction and `trap_crash_report.txt` in the TRAP folder for TRAP
([Output products](../concepts/products.md)). For the steps leading up to the error,
read `reduction.log` or `trap_reduction.log` in the same folder.

## Follow a running reduction

Each observation writes a readable log, `reduction.log`, and the same records as JSON
lines, `reduction.jsonlog`, which the tools above read. To watch one reduction, follow its log.

```bash
tail -f "$HOME/data/sphere/reduction/IRDIS/observation/*_51_Eri/DB_K12/2015-09-24/reduction.log"
```

The quotes matter, because `*` is part of the target folder name.

## All TRAP results at a glance

`plot_trap_mosaics` draws the detection map and the companion spectrum of every
observation below a TRAP directory into one figure per template.

```bash
plot_trap_mosaics ~/data/sphere/reduction/IRDIS/trap --format png
```

It writes `combined_mosaic_flat.png`, `combined_mosaic_L-type.png` and
`combined_mosaic_T-type.png` into a `mosaics/` folder below the given directory, or
into `--output`. With `$SPHERICAL_DATABASE_DIR` set, or `--database-dir`, the panel
titles add the exposure time, field rotation and seeing from the database. For large
surveys, `--batch-size` splits the figure into several files, and `--snr-min` hides
weak candidates. All options are on [Command-line tools](../reference/cli.md).

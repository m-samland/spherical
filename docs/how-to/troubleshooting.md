# Troubleshooting and FAQ

Find the cause of a common problem and what to do about it. Problems with the
installation itself are covered on [Installation](../getting-started/installation.md).

## Database queries

### A star name gives a warning about SIMBAD

`filter(target_list=[...])` looks the name up in the tables first. For a name it does
not find there, it warns
`Target '<name>' not found in local ID columns. Trying SIMBAD...` and asks SIMBAD,
which needs a network connection. If SIMBAD knows the star but it was never observed,
the next warning says that its `MAIN_ID` is not in the observation table. If SIMBAD does
not know the name either, the warning is `SIMBAD could not resolve '<name>'.` Check the
spelling, or search the `MAIN_ID` and `ID_*` columns yourself.

### `KeyError: '<name>' is not a column`

A keyword passed to `filter` must be a column of the observation table. The message
suggests the closest column name when there is one, and lists all columns.
`db.columns` lists them too.

## Starting a reduction

### `ModuleNotFoundError: No module named 'charis'` or `'trap'`

The reduction needs the `pipeline` extra, which the database-only installation leaves
out. Install it as described on [Installation](../getting-started/installation.md).

### `ValueError: IRDIS observation received an IFSReductionConfig`

Each instrument needs its own configuration class. Pass an `IRDISReductionConfig` for
IRDIS observations and an `IFSReductionConfig` for IFS ones, or `config=None` for the
defaults. The same error exists the other way round. Reduce the two instruments with
separate calls to `execute_targets`.

### `ValueError: Unknown step name(s) in force`

A name in `config.steps.force` is not a step of the instrument. The message lists the
valid names, and nothing runs. [Re-run part of a reduction](rerun.md) lists the step
names and the common cases.

### ESO asks for a password in every run, or a batch job hangs at the download

The download logs in to ESO for proprietary data and asks for the password in the
terminal unless it is stored in the keyring. [Proprietary data and ESO
credentials](proprietary-data.md) explains how to store it once.

## During and after a reduction

### The log says `outputs present` and the step does nothing

This is {term}`Resume`. The step finished in an earlier run. To compute it again with
new settings, force it ([Re-run part of a reduction](rerun.md)).

### The aligned cube did not change after I edited `config.alignment`

`align_frames` is marked done by a hidden file, `.align_frames.done`, and is not rerun
because a setting changed. Force it with `force={"align_frames"}`.

### The star position looks wrong or has gaps

Look at the plots in `converted/center_plots/`, which show the fitted positions over
the sequence. The positions TRAP uses are in `image_centers_fitted_robust.fits`. How
they are made depends on the instrument and on whether the waffle spots stay on
([Anatomy of a reduction](../concepts/anatomy.md)). `align_frames` leaves a frame
without a finite centre empty (NaN). A sequence without {term}`CENTER frame`s cannot be centred, which
is one reason it is not {term}`HCI_READY`.

### `/dev/shm` is full after a TRAP run was killed

A killed TRAP run leaves its working data behind. Delete it with
`rm -rf /dev/shm/trap_store_*` and see [Run on a server or cluster](server.md) for a
safer location.

### Where is the error message of a failed reduction?

In `crash_report.txt` in the observation folder, or `trap_crash_report.txt` in the TRAP
folder. [Monitor runs](monitor.md) shows how to collect them for many targets.

## Reporting a problem

Open an issue on [GitHub](https://github.com/m-samland/spherical/issues) with the bug
report template. Include the crash report, the end of `reduction.log`, your
configuration and the output of `pip show spherical`.

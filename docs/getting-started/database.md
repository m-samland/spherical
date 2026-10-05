# Get the database

After this page you have the observation tables on disk, and spherical knows where
they are. Each row of an observation table is one {term}`Observation sequence` in the
ESO archive. The tables hold metadata only, no reduced data.

## Download the tables

The published tables are on [Zenodo](https://doi.org/10.5281/zenodo.15147730).
`spherical-sync-tables` fetches the latest release.

```bash
spherical-sync-tables --dest ~/data/sphere/database
```

Downloads are checked against their md5 sums and resume when interrupted. Tables you
updated or regenerated locally are kept unless you pass `--force`. Add `--list` to
see what would be fetched without downloading. The IRDIS polarimetry and the sparse
aperture masking tables are left out unless you add `--include-polarimetry` or
`--include-sam`.

Release v3.0.0 (Zenodo record 21889452, ESO archive up to 2026-08-09) is 265 MB on
disk, most of it in the two file tables.

## What you get

`table_of_observations_ifs.fits`, `table_of_observations_irdis.fits`
: One row per observation sequence, with the target, instrument setup, conditions,
  quality flags and the Gaia and MOCA information about the star. These are the
  tables you search.

`table_of_files_ifs.csv`, `table_of_files_irdis.csv`
: One row per raw file in the archive. spherical uses them to find the files of a
  sequence and its calibrations.

`table_of_targets_ifs.fits`, `table_of_targets_irdis.fits`
: One row per star, with its catalogue identifiers and enrichment.

`database_provenance.json`
: How and when each table was built, and the archive period it covers.

## Tell spherical where the tables are

Set `$SPHERICAL_DATABASE_DIR` once, for example in `~/.zshrc` or `~/.bashrc`.

```bash
export SPHERICAL_DATABASE_DIR=~/data/sphere/database
```

Every entry point reads it.

| Entry point | Without the variable |
|---|---|
| `spherical-sync-tables`, `spherical-update-database` | `--dest` is required |
| `examples/ifs_reduction_template.py`, `examples/irdis_reduction_template.py` | fall back to `~/data/sphere/database` |
| `examples/explore_database.ipynb` | falls back to `~/data/sphere/database`; its `database_dir` setting overrides both |
| `plot_trap_mosaics` | uses `--database-dir`; without either, figure titles omit exposure time and rotation |

A directory you give explicitly, as a command-line option or in your own code,
always wins over the variable. In your own scripts,
`spherical.database.paths.resolve_database_dir(explicit=None, default=None)` applies
the same order: the explicit directory, then the variable, then the default.

## Bring the tables up to date

The published tables end at their release date. `spherical-update-database` extends
them to today from the ESO archive, rebuilds the target and observation tables for
every mode, and runs the Gaia DR3 and MOCA enrichment.

```bash
spherical-update-database --dest ~/data/sphere/database
```

To repeat only the enrichment, for example after a catalogue update, without
querying ESO, run

```bash
spherical-update-database --dest ~/data/sphere/database --enrich-only --mode irdis
```

The command prints a health summary of the enrichment for each mode and exits with
an error if an enrichment failed or got noticeably worse than in the previous run.
All options are listed under [Command-line tools](../reference/cli.md).

You can also download the tables by hand from Zenodo. You need at least the file and
observation tables of your instrument.

Next: [Your first query](first-query.md)

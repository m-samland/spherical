# Update the database to today

After this guide you can extend the observation tables with the newest archive data and
check that the update worked. [Get the database](../getting-started/database.md)
introduces the command, and [The observation database](../concepts/database.md)
explains what it builds.

## Before you start

The update runs where your tables are and writes into the same directory, so keep a
copy of the published tables if you want to go back to them.

It needs access to four services, the ESO archive for the file headers, SIMBAD for the
stars, the Gaia archive and the MOCA database. MOCA is reached over its MySQL port,
3306 (`mocadb_matching.py`). A firewall that blocks this port makes the MOCA
enrichment fail while the rest succeeds.

## Run the update

```bash
spherical-update-database --dest ~/data/sphere/database
```

For each instrument the command starts at the last night the tables cover, as recorded
in `database_provenance.json`, minus seven days of overlap (`--overlap-days`), and ends
today (`--end-date`). It extends the file table, rebuilds the target and observation
tables of every mode, enriches them and updates the provenance (`build.update_database`).

Useful options:

`--instrument ifs` or `--instrument irdis`
: Update one instrument only.

`--skip-sam`
: Leave out the sparse aperture masking tables.

`--no-enrich`
: Skip the Gaia and MOCA queries, for example when one of them is down. Run
  `--enrich-only` later.

`--start-date YYYY-MM-DD`
: Start somewhere else than the recorded coverage.

All options are on [Command-line tools](../reference/cli.md).

```{warning}
Do not use `--suffix` for a trial build next to your real tables. The suffixed build
writes its provenance under the same mode names as the real tables, so it replaces
their entries in `database_provenance.json`. Copy the tables to another directory and
point `--dest` there instead.
```

## Check the result

At the end the command prints a health summary of the enrichment for every mode it
processed. This run repeated only the enrichment of the IFS tables on a copy of the
v3.0.0 tables, which took 3 minutes:

```bash
spherical-update-database --dest ~/data/sphere/database_copy --enrich-only --mode ifs
```

```text
=== enrichment health summary ===
  ifs                  gaia    71%  OK
  ifs                  moca    87%  OK
```

The percentages are the share of targets with a Gaia or MOCA match. An enrichment
fails the check when its query failed, when its share is below a floor (40 % for Gaia,
50 % for MOCA), or when it dropped by more than 10 % compared with the previous run
(`enrichment_health.py`). The command then names the reason and exits with status 1,
so a script or a scheduled job can notice it. Repeat a failed enrichment with
`--enrich-only --mode <mode>`, without querying ESO again.

## Update on a schedule

A weekly update from `cron`, writing to a log file, could look like this:

```text
0 6 * * 1  SPHERICAL_DATABASE_DIR=$HOME/data/sphere/database $HOME/.local/bin/spherical-update-database >> $HOME/spherical-update.log 2>&1
```

Use the path where your environment installed `spherical-update-database`
(`which spherical-update-database` shows it).

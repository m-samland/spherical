# The big picture

After reading this page you know the path your data takes from the ESO archive to the
files you analyse, and which page explains each stage.

```{pipeline-map}
:archive: concepts/database
:database: concepts/database
:reduction: concepts/anatomy
:trap: concepts/adi-sdi-trap
:products: concepts/products
```

## Two halves

spherical has a database and a pipeline. The database holds one row per
{term}`Observation sequence` in the ESO archive, with the target, the instrument
setup, the observing conditions, quality flags and properties of the star. It holds
metadata only. You download it once ([Get the database](../getting-started/database.md))
and search it with `SphereDatabase` ([Your first query](../getting-started/first-query.md)).
[The observation database](database.md) explains how the rows are made.

The pipeline turns the sequences you select into science products. You run it from a
Python script that calls two functions in order, as both reduction templates do.
`execute_targets(observations, config)` downloads and reduces each sequence, and
`run_trap_on_observations(observations, trap_config, reduction_config, species_database_directory)`
runs {term}`TRAP` on the reduced data. There is no command-line tool for reductions.

## What you configure

A reduction script builds three things.

- The observations, a list made by `SphereDatabase.retrieve_observation_metadata`
  from the table rows you selected.
- The reduction configuration, an `IFSReductionConfig` or an `IRDISReductionConfig`.
  [The configuration model](configuration.md) explains how to change it.
- The TRAP configuration from `trap_config_for_ifs()` or `trap_config_for_irdis()`,
  described on [TRAP settings](../reference/trap-settings.md).

## What runs where

Queries run on the tables on your disk. A network connection is needed only to
download raw data from the archive, to look up a star name that is not in the
tables (spherical then asks SIMBAD), and to update the tables. The reduction runs on
your machine and writes everything below the directories in `config.directories`.
[Output products](products.md) shows the layout, and
[Run on a server or cluster](../how-to/server.md) covers long unattended runs.

Next: [SPHERE in five minutes](sphere.md)

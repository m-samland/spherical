# Command-line tools

These commands are installed with spherical. They manage the database tables and
summarise finished reductions; the reductions themselves run from Python scripts.

## spherical-sync-tables

```{eval-rst}
.. argparse::
   :module: spherical.scripts.sync_zenodo_tables
   :func: build_parser
   :prog: spherical-sync-tables
```

`spherical-sync-zenodo-tables` is an alias of the same command.

## spherical-update-database

```{eval-rst}
.. argparse::
   :module: spherical.scripts.update_database
   :func: build_parser
   :prog: spherical-update-database
```

## reduction_status

```{eval-rst}
.. argparse::
   :module: spherical.scripts.aggregate_reduction_status
   :func: build_parser
   :prog: reduction_status
```

## crash_reports

```{eval-rst}
.. argparse::
   :module: spherical.scripts.aggregate_crash_reports
   :func: build_parser
   :prog: crash_reports
```

## plot_trap_mosaics

```{eval-rst}
.. argparse::
   :module: spherical.scripts.plot_trap_mosaics
   :func: build_parser
   :prog: plot_trap_mosaics
```

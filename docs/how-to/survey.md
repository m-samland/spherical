# Reduce a list of targets or a survey

After this guide you can select many sequences from the database and reduce them with
one script. The outputs come from the v3.0.0 IRDIS tables.

## Select by a list of stars

Put the stars in a text file, one name per line. Blank lines are skipped and
everything after a `#` is a comment.

```text
51 Eri
HD 95086   # a second host

* bet Pic
```

Read it with `read_host_list` and pass the names to `filter`, together with any quality
cuts. A tuple `(op, value)` compares a column, and a list keeps the rows whose value is
in it.

```python
from astropy.table import Table

from spherical.database.multi_epoch_filter import read_host_list
from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase

database_dir = resolve_database_dir(default="~/data/sphere/database")
db = SphereDatabase(
    Table.read(database_dir / "table_of_observations_irdis.fits"),
    Table.read(database_dir / "table_of_files_irdis.csv"),
    instrument="irdis",
)

hosts = read_host_list("hosts.txt")
print(hosts)
selected = db.filter(target_list=hosts, usable_only=True, FILTER=["DB_H23", "DB_K12"], MEAN_FWHM=("<", 1.0))
print(len(selected))
print(selected["MAIN_ID", "NIGHT_START", "FILTER", "MEAN_FWHM", "ROTATION"][:5])
```

```text
['51 Eri', 'HD 95086', '* bet Pic']
28
 MAIN_ID  NIGHT_START FILTER MEAN_FWHM ROTATION
--------- ----------- ------ --------- --------
*  51 Eri  2015-09-24 DB_K12     0.758   41.502
*  51 Eri  2016-12-12 DB_H23     0.841   45.027
*  51 Eri  2017-09-27 DB_K12     0.471   52.611
*  51 Eri  2018-09-17 DB_K12     0.923   38.456
*  51 Eri  2019-11-27 DB_K12     0.486   39.469
```

`MEAN_FWHM` is the mean seeing during the sequence in arcseconds. To leave stars out
instead, pass `exclude_targets=read_host_list("exclude.txt")`.

## Select a survey sample

Without `target_list`, `filter` works on the whole table. `public=True` keeps sequences
taken more than a year ago, which are past ESO's proprietary period.

For astrometric follow-up you often want stars observed on several nights.
`select_multi_epoch_targets` keeps stars observed on at least `min_epochs` nights
whose proper motion moves a background star by at least `min_bg_motion_px` pixels
between the first and the last night. Apply the quality cuts first, because the epochs
are counted on the rows you pass in.

```python
from astropy.table import Table

from spherical.database.multi_epoch_filter import select_multi_epoch_targets
from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase

database_dir = resolve_database_dir(default="~/data/sphere/database")
db = SphereDatabase(
    Table.read(database_dir / "table_of_observations_irdis.fits"),
    Table.read(database_dir / "table_of_files_irdis.csv"),
    instrument="irdis",
)

survey = db.filter(usable_only=True, public=True, FILTER="DB_H23", MEAN_FWHM=("<", 1.0))
print(len(survey), len(set(survey["MAIN_ID"])))
multi = select_multi_epoch_targets(survey, min_bg_motion_px=1.0, min_epochs=2)
print(len(multi), len(set(multi["MAIN_ID"])))
```

```text
870 630
326 127
```

That is 870 sequences of 630 stars, of which 127 stars with 326 sequences have at
least two usable epochs.

## Reduce the selection

The reduction script is the same as for one star. Put the work inside `main()` and call
it under `if __name__ == "__main__":`. The IRDIS pre-processing starts new Python
processes, and each of them imports your script, so anything outside that guard would
run again in every process.

```python
from pathlib import Path

from astropy.table import Table
from trap.parameters import trap_config_for_irdis

from spherical.database.multi_epoch_filter import read_host_list
from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IRDISReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations


def main():
    database_dir = resolve_database_dir(default="~/data/sphere/database")
    db = SphereDatabase(
        Table.read(database_dir / "table_of_observations_irdis.fits"),
        Table.read(database_dir / "table_of_files_irdis.csv"),
        instrument="irdis",
    )
    selected = db.filter(
        target_list=read_host_list("hosts.txt"),
        usable_only=True,
        FILTER=["DB_H23", "DB_K12"],
        MEAN_FWHM=("<", 1.0),
    )
    observations = db.retrieve_observation_metadata(selected)

    config = IRDISReductionConfig()
    config.directories.base_path = Path.home() / "data/sphere"
    config.directories.raw_directory = config.directories.base_path / "data"
    config.directories.reduction_directory = config.directories.base_path / "reduction"
    config.set_ncpu(8)

    trap_config = trap_config_for_irdis()
    config.apply_trap_resources(trap_config)
    trap_config.processing = trap_config.processing.merge(use_progress_bar=False)

    execute_targets(observations=observations, config=config)
    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=config.directories.base_path / "species",
    )


if __name__ == "__main__":
    main()
```

The reduction templates set many more options, all described in the
[Configuration reference](../reference/configuration.md) and on
[TRAP settings](../reference/trap-settings.md).

## What happens with many targets

The observations are reduced one after another. If one fails, its error goes to a
crash report and the next one starts ([Anatomy of a reduction](../concepts/anatomy.md)).
When you run the script again, finished steps are skipped, so you can add stars to the
list and rerun the same script. [Monitor runs](monitor.md) shows how to check which
targets finished, and [Run on a server or cluster](server.md) how to run a long
survey unattended.

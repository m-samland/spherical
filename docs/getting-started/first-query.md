# Your first query

After this page you can load the database, find the sequences of a star and select
observations by any column. The outputs below come from the v3.0.0 tables.

## Load the database

```python
from collections import Counter

from astropy.table import Table

from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase

database_dir = resolve_database_dir(default="~/data/sphere/database")
observations = Table.read(database_dir / "table_of_observations_ifs.fits")
files = Table.read(database_dir / "table_of_files_ifs.csv")
db = SphereDatabase(observations, files, instrument="ifs")
```

`resolve_database_dir` uses `$SPHERICAL_DATABASE_DIR` when it is set and
`~/data/sphere/database` otherwise, like the reduction templates. Import
`SphereDatabase` from its module, `spherical.database.sphere_database`, because
`spherical.database` itself exports nothing.

## Find a star

```python
eri = db.filter(target_list=["51 Eri"], usable_only=True)
print(len(eri))
print(eri["MAIN_ID", "NIGHT_START", "FILTER", "OBS_PROG_ID", "OBS_ID", "TOTAL_EXPTIME_SCI", "ROTATION"][:5])
```

```text
10
 MAIN_ID  NIGHT_START FILTER  OBS_PROG_ID    OBS_ID  TOTAL_EXPTIME_SCI ROTATION
--------- ----------- ------ ------------- --------- ----------------- --------
*  51 Eri  2015-09-24  OBS_H 095.C-0298(D) 200363269            68.267   43.216
*  51 Eri  2015-09-25 OBS_YJ 095.C-0298(D) 200363328            72.533   43.141
*  51 Eri  2016-01-15 OBS_YJ 096.C-0241(G) 200374558            68.267   40.226
*  51 Eri  2016-12-09 OBS_YJ 198.C-0209(C) 200412556              19.2   11.211
*  51 Eri  2016-12-11 OBS_YJ 198.C-0209(C) 200412630              57.6   25.339
```

`target_list` takes any name SIMBAD knows. Names already in the table, such as
`51 Eri` or `HD 29391`, are found locally. Others are resolved through SIMBAD, which
needs a network connection. `usable_only=True` keeps sequences marked
{term}`HCI_READY`, taken with {term}`Pupil tracking` and with at least 5 minutes of
science exposure. `TOTAL_EXPTIME_SCI` is in minutes and `ROTATION`, the
{term}`Field rotation`, in degrees.

The first row, 2015-09-24 in `OBS_H`, is the sequence the tutorials reduce.

## Select by any column

Every column of the table can be a keyword of `filter`. A single value tests
equality and a list tests membership. A tuple `(op, value)` applies an operation,
which is a comparison such as `">"` or `"<="`, a membership test with `"in"` or
`"not in"`, or a text search with `"contains"` or `"not contains"`.

```python
programme = db.filter(OBS_PROG_ID=("contains", "095.C-0298"))
print(len(programme))
for run, count in sorted(Counter(programme["OBS_PROG_ID"]).items()):
    print(run, count)
```

```text
230
095.C-0298(A) 66
095.C-0298(B) 52
095.C-0298(C) 22
095.C-0298(D) 47
095.C-0298(G) 1
095.C-0298(H) 29
095.C-0298(I) 10
095.C-0298(J) 3
```

ESO splits a programme into runs, lettered (A), (B) and so on
({term}`Programme ID`). Matching on the text without the letter collects all of them.

## IFS and IRDIS together

In {term}`IRDIFS` mode IRDIS and IFS record the same sequence, and both tables give
it the same {term}`OBS_ID`.

```python
irdis = SphereDatabase(
    Table.read(database_dir / "table_of_observations_irdis.fits"),
    Table.read(database_dir / "table_of_files_irdis.csv"),
    instrument="irdis",
)
print(irdis.filter(OBS_ID=200363269)["MAIN_ID", "NIGHT_START", "FILTER", "OBS_PROG_ID", "OBS_ID"])
```

```text
 MAIN_ID  NIGHT_START FILTER  OBS_PROG_ID    OBS_ID
--------- ----------- ------ ------------- ---------
*  51 Eri  2015-09-24 DB_K12 095.C-0298(D) 200363269
```

```{expected-result}
With the v3.0.0 tables you get 10 usable IFS sequences of 51 Eri, the first on
2015-09-24 in `OBS_H`, and the IRDIS row of the same sequence in `DB_K12`. Later
releases can add rows. A `FileNotFoundError` means the tables are not in
`$SPHERICAL_DATABASE_DIR`, or in `~/data/sphere/database` when the variable is
not set.
```

Next: [Next steps](next-steps.md)

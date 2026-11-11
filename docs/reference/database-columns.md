---
stage: database
---

# Database columns

Look up what a column of the database tables means and which unit it is in.

The database has four kinds of table, each once for IFS and once for IRDIS
(`table_of_<kind>_ifs` and `table_of_<kind>_irdis`), with the same columns in both.

- The **observation table** has one row per {term}`Observation sequence` of a star, with
  the star's catalogue columns and a summary of the sequence.
- The **target table** has one row per star matched in SIMBAD. All its columns are
  also in the observation table.
- The **other observations table** holds the sequences set aside because they are not
  observations of a star (see [Sequences set aside](../concepts/database.md#sequences-set-aside)).
- The **file table** has one row per raw file in the ESO archive, with header values
  under shorter names.

The polarimetry and sparse aperture masking tables (`_polarimetry`, `_sam`) have the
same columns as their parent table, except the `_sam` other observations tables, which
are empty and carry 9 columns only.

To list the columns of the observation table you loaded, use `db.columns`.

```python
from spherical.database.sphere_database import SphereDatabase

db = SphereDatabase(observations, files, instrument="ifs")
db.columns
```

Only a few target columns store their unit in the FITS file (`RA_DEG`, `DEC_DEG`,
`POS_DIFF`, `POS_DIFF_ORIG`, `DISTANCE`). The other units on this page come from the
code that writes each column. Most header keywords missing from a file are stored as
-10000 for numbers and `N/A` for text, in the file table and in the observation columns
copied from it, such as `WAFFLE_AMP`. The exceptions are `OBS_NAME` (-10000), `DET_ID`
(`ZPL`), `FILE_SIZE` (0) and the FITS layout keywords, which take a typical value
(`file_table.header_list`).

## Observation table

```{eval-rst}
.. column-table:: observations
```

## Target table

```{eval-rst}
.. column-table:: targets
```

## Other observations table

```{eval-rst}
.. column-table:: other_observations
```

## File table

```{eval-rst}
.. column-table:: files
```

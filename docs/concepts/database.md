# The observation database

After reading this page you know how the observation tables are built, what one row
stands for and what its flags and enrichment columns mean.

## How the tables are built

`spherical-update-database` builds the tables in four stages
(`database/build.py`, `build_tables`).

1. **Files.** Every raw SPHERE file in the ESO archive becomes a row of the file
   table (`database/file_table.py`), `table_of_files_ifs.csv` or `table_of_files_irdis.csv`, with the values of
   its FITS header.
2. **Targets.** spherical asks SIMBAD which star lies within 3″ of the coordinates in
   the headers. It keeps stars for which SIMBAD has a J magnitude no fainter than
   14 mag, a proper motion and a positive parallax. The result is the target table.
3. **Enrichment.** Each target is matched to the MOCA database of young stars and to
   Gaia DR3 (see [Enrichment](#enrichment)).
4. **Sequences.** The science files within 15″ of a target are grouped into
   sequences, and each sequence becomes one row of the observation table.

The published tables were built this way and are refreshed with each release. The
[update guide](../how-to/update-database.md) shows how to run the same build for the
newest archive data.

## One row, one sequence

A row of `table_of_observations_*.fits` is one {term}`Observation sequence`, all
science frames of one target taken in one night (`NIGHT_START`) in one mode (`FILTER`,
from the IFS mode or the IRDIS filter pair) (`observation_table.create_observation_table`).
If a star was observed twice in the same night and mode, both blocks form one row.

`PRIMARY_SCIENCE` says which frames carry the science. It is `CORO` or `CENTER`,
whichever holds more exposure time, with `CORO` winning a tie. `FLUX` appears only for
a sequence that was stopped before any CORO or CENTER frame. `WAFFLE_MODE` is true when
the CENTER frames win, which marks a {term}`Continuous waffle` sequence
(`select_primary_science_frames`).

`ROTATION` is the {term}`Field rotation` in degrees, the difference between the
{term}`DEROT ANGLE` of the first and the last science frame. `TOTAL_EXPTIME_SCI` and
its siblings per frame type are in minutes.

## Quality flags

The flags describe what a sequence is missing or mixes (`evaluate_observation_flags`).

| Column | True when |
|---|---|
| `FLUX_FLAG`, `CENTER_FLAG`, `CORO_FLAG` | the sequence has no frames of that type |
| `CENTER_DIT_FLAG`, `CORO_DIT_FLAG` | the CENTER or CORO frames use more than one exposure time (DIT) |
| `FLUX_DIT_FLAG`, `FLUX_ND_FLAG` | the FLUX frames mix exposure times or {term}`ND filter`s |
| `DEROTATOR_FLAG` | the derotation angles could not be computed |

`FLUX_DIT_SPREAD` gives the ratio of the longest to the shortest FLUX exposure time.
Mixed FLUX setups are reported but do not block a reduction, because the pipeline
scales each FLUX frame by its own DIT and ND filter.

{term}`HCI_READY` combines the flags that matter for the pipeline. It is true when the
sequence has CENTER and FLUX frames, one DIT across its CENTER frames and one across
its CORO frames, and angles that could be computed (`compute_hci_ready`). It does not
check the derotator mode. `filter(usable_only=True)` adds {term}`Pupil tracking` and at
least 5 minutes of science exposure (`sphere_database.usable_mask`).

(enrichment)=
## Enrichment

The target columns come from SIMBAD: identifiers (`ID_HD`, `ID_HIP`, `ID_GAIA_DR3`,
...), spectral type, parallax, proper motion and magnitudes. Two catalogues add more.

Gaia DR3
: The stellar parameters from Gaia's GSP-Phot pipeline, matched by Gaia DR3 source
  ID: `GAIA_TEFF`, `GAIA_LOGG`, `GAIA_MH` and the extinction `GAIA_AG`, each with
  `_LOWER` and `_UPPER` bounds (`gaia_astrophysical_params.py`). By default the
  reduction uses them as the stellar parameters for TRAP's template matching
  (`use_gaia_stellar_parameters`, see [TRAP settings](../reference/trap-settings.md)).

MOCA
: Membership in young associations and clusters and the resulting age, from the
  Montreal Open Clusters and Associations database
  ([Gagné et al. 2026](https://arxiv.org/abs/2602.15695)). The columns start with
  `MOCA_`, for example `MOCA_ASSOCIATION_NAME` and `MOCA_AGE_MYR` with its
  uncertainties, followed by membership probabilities and youth indicators
  (`mocadb_matching.py`).

A star without a match has empty values in these columns. `filter` leaves out rows
whose value is missing for a column you filter on.

## Provenance

`database_provenance.json` records for every table the spherical version that built
it, the archive period it covers (`eso_coverage_start`, `eso_coverage_end`), when ESO,
Gaia and MOCA were queried, and how many targets each enrichment matched
(`database/provenance.py`). The same record is stored in the metadata of each table
file, so a table you copy elsewhere keeps it.

## From a row to an observation

The pipeline does not work on table rows directly.
`SphereDatabase.retrieve_observation_metadata(table)` turns each row into an
`IFSObservation` or `IRDISObservation`. It collects the science files of that night
and mode within 1.2′ of the target, which includes sky frames taken a little off the
star, and the calibration files the reduction needs, all from the file table.

## Columns

The observation table has 118 columns, grouped as target identity and
properties, Gaia and MOCA enrichment, instrument setup, timing, exposures, quality
flags, observing conditions, rotation and programme. A generated reference of every
column will follow. Until then, `db.columns` lists them and
[Your first query](../getting-started/first-query.md) shows the ones you use most.

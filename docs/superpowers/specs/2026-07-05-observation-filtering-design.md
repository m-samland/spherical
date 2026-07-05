# Observation table filtering & presentation — design

Date: 2026-07-05
Status: Approved (brainstorming), pending spec review
Repo: `spherical`, target branch: feature branch off `develop`

## Problem

Filtering the observation table is currently hand-rolled and duplicated. Two
entry points both build a raw `np.logical_and.reduce([...])` mask:

- `explore_database.ipynb` masks `table_of_observations` directly, bypassing the
  `SphereDatabase` object entirely, then `table[mask].show_in_browser()`.
- `examples/ifs_reduction_template.py` calls
  `target_list_to_observation_table(target_list)` → Table → a second
  hand-written mask → `retrieve_observation_metadata(table)`.

Pain points:

1. The filtering mechanism (`logical_and.reduce`) is duplicated and verbose in
   two places.
2. List membership (`OBS_PROG_ID` include/exclude, `MOCA_ASSOCIATION_NAME`) is
   awkward with raw numpy (`~np.isin(...)`).
3. "Usability" filtering (`usable_only`, a precomputed mask on the DB) and
   ad-hoc filtering are two unrelated mechanisms that don't compose.
4. Presentation (the `NORMAL`/`SHORT`/`MEDIUM`/`OBSLOG` column-sets) is welded
   into `show_in_browser` and the SIMBAD lookups, so a filtered table cannot
   reuse a named summary without copying the column list.
5. **Latent correctness bug:** many columns are `MaskedColumn`s with real
   missing values (`MOCA_AGE_MYR`, `GAIA_TEFF`, `MOCA_ASSOCIATION_NAME`, …). A
   naive comparison such as `table['MOCA_AGE_MYR'] < 50` returns a *masked*
   boolean; when used to index, astropy falls back to the underlying fill value,
   so missing-value rows are included or excluded by an arbitrary fill rather
   than by the criterion. The result is effectively undefined and silent.

The motivating new capability is include/exclude filtering by lists of
`OBS_PROG_ID`. The guiding constraint is minimalism: solve the real issues
without inventing a query DSL or bloating the class.

## Current state (findings)

- Interchange object is an astropy `Table` in both flows;
  `retrieve_observation_metadata(table)` consumes a `Table`.
- `SphereDatabase` already has: a precomputed `_not_usable_observations_mask`
  (HCI_READY + not FIELD + min exposure) exposed via `usable_only=`; the
  summary column-sets built in `__init__`; robust name resolution
  (`_find_observations_by_id_or_simbad`, with binary-naming and SIMBAD
  fallback) used by `observations_from_name_SIMBAD` /
  `get_observation_SIMBAD` / `retrieve_observation`.
- Missing values in the shipped table are encoded as **masked entries in
  `MaskedColumn`s**. `np.ma.getmaskarray(col)` reads the mask uniformly
  (all-present for unmasked columns like `FLUX_J`, `DISTANCE`).
- The build machinery writes **plain `Column`s with raw placeholders**, not
  masked columns: Gaia and MOCA floats use `np.nan`, MOCA strings and target IDs
  use `""` / `np.nan`. The astropy **FITS write→read cycle silently converts
  `np.nan` and `""` to masks**, so on read these all become proper masked
  columns — verified on the shipped table (`GAIA_*`, `MOCA_*`, `ID_HD`,
  `ID_HIP` all masked). The one exception is a numeric sentinel that does *not*
  round-trip to a mask: `ROTATION = -10000` (see the source-fix decision).
- pandas is a base dependency, but the design deliberately stays in
  numpy/astropy — no query-string path (rejected, see below).

## Decisions

### Filter expression style — validated kwargs + numpy mask escape hatch

Rejected: a pandas `query()` string (silent `=` footgun, cryptic messages, no
column validation) and a full `(op, value)` predicate grammar (reinvents numpy
comparisons for no gain). Chosen: a small hybrid.

- `**criteria` — three self-evident rules, nothing more:
  - scalar → equality (`MOCA_ASSOCIATION_NAME='beta Pictoris moving group'`)
  - list → membership (`OBS_PROG_ID=['110.240D.001', '111.24QL.001']`)
  - `('not in', list)` → exclusion (`OBS_PROG_ID=('not in', ['095.C-0298'])`)
  Each key is validated against `colnames`; unknown → `KeyError` naming it, with
  a `difflib` suggestion and the available-column list.
- `*masks` — boolean arrays or `callable(table) -> bool array`, AND-combined.
  All comparisons and **cross-column arithmetic** (absolute magnitude
  `FLUX_J - 5*log10(DISTANCE/10)`, colors `FLUX_J - FLUX_H`, relative age error
  `MOCA_AGE_MYR_UNC / MOCA_AGE_MYR`) live here, in the plain numpy the user
  already writes.

### Return type — a filtered `Table`

`filter()` returns a copied astropy `Table`, not a chainable DB view. It is the
most intuitive (a Table is what it looks like), the most maintainable (pure
function of the table; no cheap-copy constructor, no `__init__` re-run of
masks/lookups/calibration), and matches the existing interchange exactly.
Chaining is served by passing multiple criteria in one call. Empty result →
empty table, not an error.

### Presentation — a standalone `view()` function

The `NORMAL`/`SHORT`/`MEDIUM`/`OBSLOG` column-sets move out of `__init__` into
one module-level `SUMMARY_COLUMNS` dict. `view(table, summary=None) -> Table`
selects the subset and is pure (depends only on the table passed in), so it
works on any table — filtered or not. It intersects the requested columns with
`table.colnames`, silently dropping absent ones (preserving today's behavior).
With `FAILED_SEQ` removed (see below), the readiness column is simply
`HCI_READY` — no `{READY}` placeholder indirection remains. `show_in_browser`
and the `summary=` returns in the SIMBAD lookups become thin calls to `view()`.
No duplicated column lists remain.

### Missing values — a criterion excludes rows where its column is missing

Fixed, non-optional semantics (documented, not a flag): if a row lacks a value
for a column you filter on, the criterion cannot be asserted, so the row is
excluded.

- For `**criteria`: for each criterion on column `C`, explicitly AND in
  `~np.ma.getmaskarray(tbl[C])`. This is the guarantee, and it covers the
  `not in` case that `filled()` alone cannot (numpy's `isin` drops the mask).
- For `*masks`: `filter()` runs `np.ma.filled(final_mask, False)` before
  indexing. This auto-excludes missing rows for all comparison-based masks
  (including cross-column ones), since a comparison against a masked operand
  yields a masked boolean. A user hand-writing `isin` inside a lambda owns their
  own presence handling; documented. No `present()` helper is added — `filled`
  covers comparisons and validated kwargs cover membership.

The rare "I want the missing ones" case is served explicitly via
`lambda t: np.ma.getmaskarray(t['GAIA_TEFF'])`.

### `target_list` folded into `filter()`

`filter()` takes an optional `target_list`. When given, it **reuses** the
existing name-resolution machinery (`_find_observations_by_id_or_simbad`: local
ID lookup, binary-naming variants, SIMBAD fallback) to resolve names to
`MAIN_ID`s, restricts the table to those, then applies masks/criteria/
`usable_only`/presence. When `None`, it operates on the whole table. This
collapses the reduction script's `target_list_to_observation_table(...)` +
hand-written mask into a single call, unifying both entry points on one method.
`obs_band` / `date` become ordinary criteria (`FILTER='OBS_H'`,
`NIGHT_START=...`).

### `usable_only` — keep the flag distinction, consolidate the machinery

`HCI_READY` is **not** a duplicate of `usable_only`; it is one of its three
ingredients. Measured on the current IFS table (4726 rows): `HCI_READY` is True
for 3668, `usable_only` for 3071. The 597-row gap is real —
`HCI_READY == True` but not usable breaks down as 176 `DEROTATOR_MODE == 'FIELD'`
(data-complete but field-stabilized, useless for ADI) and ~489
`TOTAL_EXPTIME_SCI < 5s`. `HCI_READY` = "does this sequence have the
frames/DITs/derotator data it needs?"; `usable_only` = that **and**
pupil-stabilized **and** long enough. So `usable_only` stays.

But its implementation is currently a parallel subsystem (a precomputed
`_not_usable_observations_mask`, the `_mask_not_usable_observations()` method,
and a magic `5.0` in `__init__`). With `filter()`, usability is just a named
bundle of three ordinary criteria. Consolidate it:

- One module-level `usable_mask(table) -> np.ndarray` — the single source of
  truth: `HCI_READY == True` AND `DEROTATOR_MODE != 'FIELD'` AND
  `TOTAL_EXPTIME_SCI >= USABLE_MIN_EXPTIME_SCI` (a named constant, `5.0`).
- `filter(usable_only=True)` ANDs it in; `show_in_browser` and the SIMBAD
  lookups route their `usable_only=` through the same helper.
- Drop the precomputed `_not_usable_observations_mask` and
  `_mask_not_usable_observations()`. Behavior is identical.
- Document the `HCI_READY` vs `usable_only` distinction in
  `explore_database.ipynb`.

### Remove the deprecated `FAILED_SEQ` path

`FAILED_SEQ` is the pre-`HCI_READY` readiness flag and is fully deprecated; no
shipped table still uses it. Its only remaining footprint is the `_ready_flag`
indirection in `sphere_database.py` (4 spots: the `HCI_READY`/`FAILED_SEQ`
selection at init, the `{READY}` summary placeholder, and the two-branch
`mask_bad_flag`). Remove it entirely: use `HCI_READY` directly, delete
`_ready_flag`, and drop the `{READY}` placeholder from the summary-column
handling. No tests or examples reference it, so the change is confined to one
file.

### Missing-value representation — fix the one non-masked sentinel at the source

The filter's missing-value semantics assume "missing == masked." That holds for
every enrichment column (Gaia/MOCA/IDs round-trip `np.nan`/`""` to masks), with
one exception: `ROTATION = -10000`, emitted in `observation_table.py` (two
spots: polarimetry, and the failed-angle `except`). Being `-10000.0` rather than
NaN, it survives the FITS round-trip as a live numeric value the filter cannot
recognize as missing. On the current table this is 22 of 4726 rows — all with
`DEROTATOR_FLAG == True` and `HCI_READY == False`, so `usable_only` already
excludes them; only a bare `ROTATION` filter without `usable_only` is exposed.

Decision — fix at the source, add no runtime sentinel handling:

- Change `ROTATION`'s `-10000` → `np.nan` in `observation_table.py` (both
  spots). It then round-trips to a mask like everything else, making the
  observation table's missing-value representation fully uniform.
  `DEROTATOR_FLAG` already carries the "computation failed / not applicable"
  signal, so no information is lost.
- **No** load-time sentinel machinery on the observation table (no blanket
  `-10000` sweep — a legitimate value could be `-10000`; no ROTATION-specific
  runtime mask for 22 already-excluded rows).
- The fix takes effect on the next database rebuild + Zenodo re-release.
  Residual until then: pre-rebuild tables carry 22 non-usable `-10000` ROTATION
  rows; documented, low impact.

## API surface (all in `database/sphere_database.py`)

```python
SUMMARY_COLUMNS: dict[str, list[str]]        # module-level; the named sets
USABLE_MIN_EXPTIME_SCI: float = 5.0          # module-level named constant

def usable_mask(table: Table) -> np.ndarray:
    """HCI_READY & DEROTATOR_MODE != 'FIELD' & TOTAL_EXPTIME_SCI >= min."""

def view(table: Table, summary: str | None = None) -> Table:
    """Return `table` reduced to a named column-set; None → all columns."""

class SphereDatabase:
    @property
    def columns(self) -> list[str]:          # == table_of_observations.colnames
        ...

    def filter(
        self,
        *masks,                              # bool arrays or callable(table)->bool
        usable_only: bool = False,
        target_list: list[str] | None = None,
        **criteria,                          # scalar | list | ('not in', list)
    ) -> Table:
        ...
```

Worked example (carried in the docstring):

```python
import numpy as np

db.filter(
    lambda t: t['MEAN_FWHM'] < 1.2,                               # single-column threshold (observing conditions)
    lambda t: t['FLUX_J'] - 5 * np.log10(t['DISTANCE'] / 10) < 4, # absolute J mag: genuine cross-column, needs a mask
    usable_only=True,                                            # HCI-ready + pupil-stabilized + min exposure
    MOCA_ASSOCIATION_NAME='beta Pictoris moving group',          # equality
    OBS_PROG_ID=('not in', ['095.C-0298', '096.C-0241']),        # program-ID exclusion
)
```

The masks are AND-combined: a plain numeric threshold and a cross-column
arithmetic cut (absolute J magnitude — expressible *only* as a mask, since it
combines `FLUX_J` and `DISTANCE`). Equality and membership go through the
validated kwargs.

## Data flow after the change

Notebook (explore):

```python
obs = db.filter(usable_only=True,
                MOCA_ASSOCIATION_NAME='beta Pictoris moving group',
                OBS_PROG_ID=('not in', ['095.C-0298']))
view(obs, 'OBSLOG').show_in_browser(jsviewer=True)
```

Reduction template:

```python
obs = db.filter(lambda t: t['TOTAL_EXPTIME_SCI'] > 30,   # mask is positional (*masks)
                target_list=['* bet Pic'], usable_only=True)
observations = db.retrieve_observation_metadata(obs)     # needs full columns
```

`filter()` returns full columns (never `view()`-reduced) because
`retrieve_observation_metadata` needs `MAIN_ID` / `FILTER` / `NIGHT_START`.

## Error handling

- Unknown criteria column → `KeyError` naming it + `difflib` suggestion + the
  available-column list.
- Mask of wrong length → clear `ValueError`.
- `('not in', x)` where `x` is not a list/sequence → `TypeError` with guidance.
- No name in `target_list` resolves → empty table (consistent with
  `target_list_to_observation_table`).

## Testing (database logic is CI-covered per CLAUDE.md; built test-first)

- Each criteria rule: scalar equality, list membership, `('not in', list)`.
- Mask + criteria composition; `usable_only` interaction.
- `usable_mask()` isolates the three conditions correctly (a `HCI_READY` FIELD
  row and a sub-threshold-exposure row are both excluded); `usable_only=True`
  matches `usable_mask()` applied to the same table.
- **Missing-value semantics** (the core correctness case): a masked-column
  criterion excludes missing rows for `<`, `==`, membership, and `not in`;
  a comparison mask excludes missing rows via `filled(False)`.
- `observation_table.py` emits `np.nan` (not `-10000`) for `ROTATION` when angle
  computation is skipped/fails; the resulting column round-trips to a masked
  column that the filter's presence logic excludes.
- Unknown-column error carries a suggestion.
- `view()` column selection, incl. silent drop of absent columns and the
  ready-flag placeholder.
- `target_list` restriction (local ID hit; SIMBAD fallback mocked as in
  existing tests).

## Relationship to existing methods / migration

- Name-resolution internals (`_find_observations_by_id_or_simbad`) are shared,
  not duplicated. `observations_from_name_SIMBAD` /
  `get_observation_SIMBAD` / `retrieve_observation` are unchanged (internal
  object-construction path).
- `target_list_to_observation_table` becomes redundant for user scripts (its
  job is `filter(target_list=...)`). It is **kept** for now — no drive-by
  removal on a released package — and flagged as a later deprecation candidate.
- `show_in_browser` and the SIMBAD `summary=` returns are re-pointed at
  `view()`; no behavior change.
- `usable_only=` machinery is consolidated into `usable_mask()`; the precomputed
  `_not_usable_observations_mask` and `_mask_not_usable_observations()` are
  removed. Behavior unchanged.
- The deprecated `FAILED_SEQ` / `_ready_flag` path is removed (confined to
  `sphere_database.py`).
- `observation_table.py` `ROTATION` sentinel `-10000` → `np.nan` (both spots).
  Effective on the next database rebuild + Zenodo re-release; the enrichment
  columns already round-trip to masks, so after this the obs table's
  missing-value representation is uniform.
- `explore_database.ipynb` and `examples/ifs_reduction_template.py` are updated
  to the new calls; the notebook gains a short `HCI_READY` vs `usable_only`
  explanation.

## Non-goals

- No pandas query-string path; no `(op, value)` operator grammar.
- No chainable DB view.
- No `present()` helper.
- No runtime sentinel handling on the observation table (no blanket `-10000`
  sweep, no ROTATION-specific mask-on-load). The only sentinel is fixed at the
  source.
- No editor autocomplete on column names (columns are runtime FITS data;
  impossible for any approach). Discoverability is served by `db.columns` and
  the suggestion in the error message.
- No removal of `target_list_to_observation_table` in this change.

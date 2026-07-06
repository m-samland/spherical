# Observation Table Filtering & Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `SphereDatabase` a single `filter()` entry point for the observation table (validated equality/membership kwargs + numpy mask escape hatch), a standalone `view()` for column-set presentation, uniform exclude-missing semantics, and consolidate the `usable_only` machinery — replacing the hand-rolled `np.logical_and.reduce` masks duplicated across the notebook and reduction template.

**Architecture:** All new code lives in `src/spherical/database/sphere_database.py` as module-level functions (`usable_mask`, `view`, `SUMMARY_COLUMNS`, criterion helpers) plus a `filter()` method and `columns` property. `filter()` returns a filtered astropy `Table` (the existing interchange object). Missing values are excluded uniformly: `~np.ma.getmaskarray(col)` per criterion, `np.ma.filled(mask, False)` for masks. The deprecated `FAILED_SEQ`/`_ready_flag` path and the precomputed usable-mask subsystem are removed; a one-line source fix in `observation_table.py` makes the last non-masked sentinel (`ROTATION = -10000`) round-trip to a mask.

**Tech Stack:** Python, numpy (incl. `numpy.ma`), astropy Tables, pytest, pixi.

## Global Constraints

- Work on branch `feature/observation-filtering` (already created off `develop`).
- **Run everything through the pixi `dev` environment.** Use `pixi run -e dev <cmd>` for all commands. Concretely: tests = `pixi run -e dev pytest tests/... -v`; lint = `pixi run -e dev ruff check src/...`; ad-hoc Python = `pixi run -e dev python ...`. Never use bare `python`/`pip`.
- **Commit once per task**, at the task's final step — not per sub-step. Plain commit messages, **no attribution footers** (no `Co-authored-by`, no "Generated with…" trailer).
- The `spherical.database` half MUST import without the `pipeline` extra — no `scipy`/`charis`/`trap`/`photutils` imports in this code.
- Style: ruff line-length **127**, **double quotes**, Google-style docstrings. Lint touched files before committing.
- Comment only *why*, never *what* — name things well (per CLAUDE.md).
- Tests live in `tests/`. Every behavior change lands or updates a test.
- CHANGELOG discipline: add bullets under the right `[Unreleased]` subsection in `CHANGELOG.md`, Keep-a-Changelog format.
- Never `git add -A`/`.`; stage explicit paths only.
- Missing-value semantics are **fixed** (exclude-missing), not a flag.

---

## File Structure

- **Modify** `src/spherical/database/sphere_database.py` — add `USABLE_MIN_EXPTIME_SCI`, `SUMMARY_COLUMNS`, `usable_mask()`, `view()`, `_isin()`, `_criterion_mask()`, `SphereDatabase.filter()`, `SphereDatabase.columns`; remove `_ready_flag`, `_mask_not_usable_observations()`, `_not_usable_observations_mask`, `_keys_for_*`, the local `_keys()` helper; repoint `return_usable_only`, `show_in_browser`, and the SIMBAD-lookup summary/usable branches to `view()`/`usable_mask()`.
- **Modify** `src/spherical/database/observation_table.py:154,164` — `ROTATION` sentinel `-10000` → `np.nan`.
- **Create** `tests/test_filter.py` — unit tests for `usable_mask`, `view`, criterion helpers, `filter`.
- **Modify** `tests/test_observations_from_name.py` — drop the obsolete `_not_usable_observations_mask` fixture line.
- **Modify** `tests/test_build_tables.py` — add the ROTATION-missing test (or a new focused test in `tests/test_filter.py`; see Task 6).
- **Modify** `examples/ifs_reduction_template.py` — use `filter(target_list=...)`.
- **Modify** `explore_database.ipynb` — use `filter()` + `view()`; add an `HCI_READY` vs `usable_only` note.

---

## Task 1: `usable_mask()` + `USABLE_MIN_EXPTIME_SCI`

**Files:**
- Modify: `src/spherical/database/sphere_database.py` (add module-level constant + function near the top, after the existing `_normalize_name` function)
- Test: `tests/test_filter.py`

**Interfaces:**
- Produces: `USABLE_MIN_EXPTIME_SCI: float` and `usable_mask(table: Table) -> np.ndarray` — boolean mask, True where an observation is usable (`HCI_READY` True AND `DEROTATOR_MODE != 'FIELD'` AND `TOTAL_EXPTIME_SCI >= USABLE_MIN_EXPTIME_SCI`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_filter.py`:

```python
"""Unit tests for observation-table filtering, presentation, and usable mask.

These construct small astropy Tables directly, or a lightweight SphereDatabase
via ``__new__`` (bypassing the heavy, network-dependent ``__init__``), so no
real data or network access is needed.
"""
import numpy as np
import numpy.ma as ma
import pytest
from astropy.table import Table

from spherical.database.sphere_database import (
    USABLE_MIN_EXPTIME_SCI,
    usable_mask,
)


def _usable_table():
    return Table(
        {
            "HCI_READY": [True, True, True, False],
            "DEROTATOR_MODE": ["PUPIL", "FIELD", "PUPIL", "PUPIL"],
            "TOTAL_EXPTIME_SCI": [100.0, 100.0, 1.0, 100.0],
        }
    )


def test_usable_mask_requires_all_three_conditions():
    t = _usable_table()
    result = usable_mask(t)
    # row0 usable; row1 FIELD; row2 too short; row3 not HCI_READY
    assert list(result) == [True, False, False, False]


def test_usable_min_exptime_default():
    assert USABLE_MIN_EXPTIME_SCI == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: FAIL with `ImportError: cannot import name 'USABLE_MIN_EXPTIME_SCI'`.

- [ ] **Step 3: Write minimal implementation**

In `src/spherical/database/sphere_database.py`, after the `_normalize_name` function (around line 74), add:

```python
USABLE_MIN_EXPTIME_SCI: float = 5.0


def usable_mask(table: Table) -> np.ndarray:
    """Return a boolean mask of observations usable for high-contrast imaging.

    An observation is usable when it is HCI-ready, pupil-stabilized (ADI), and
    exceeds a minimum total science exposure time.

    Parameters
    ----------
    table : astropy.table.Table
        Observation table containing ``HCI_READY``, ``DEROTATOR_MODE`` and
        ``TOTAL_EXPTIME_SCI`` columns.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the observation is usable.
    """
    ready = np.asarray(table["HCI_READY"], dtype=bool)
    pupil = np.asarray(table["DEROTATOR_MODE"] != "FIELD")
    long_enough = np.asarray(table["TOTAL_EXPTIME_SCI"] >= USABLE_MIN_EXPTIME_SCI)
    return ready & pupil & long_enough
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/spherical/database/sphere_database.py tests/test_filter.py
git commit -m "feat(database): add usable_mask() and USABLE_MIN_EXPTIME_SCI"
```

---

## Task 2: `SUMMARY_COLUMNS` + `view()`

**Files:**
- Modify: `src/spherical/database/sphere_database.py` (add module-level dict + function after `usable_mask`)
- Test: `tests/test_filter.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SUMMARY_COLUMNS: dict[str, list[str]]` (keys `"NORMAL"`, `"SHORT"`, `"MEDIUM"`, `"OBSLOG"`) and `view(table: Table, summary: str | None = None) -> Table`. `view` returns a copy reduced to the named column set (intersected with `table.colnames`, preserving that order); `summary=None` returns all columns; an unknown `summary` raises `KeyError`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_filter.py`:

```python
from spherical.database.sphere_database import SUMMARY_COLUMNS, view


def _obs_table():
    return Table(
        {
            "MAIN_ID": ["a", "b"],
            "NIGHT_START": ["2020-01-01", "2020-01-02"],
            "FILTER": ["OBS_YJ", "OBS_H"],
            "HCI_READY": [True, False],
            "OBS_PROG_ID": ["095.C-0298", "111.24QL.001"],
            "MEAN_FWHM": [0.8, 1.5],
        }
    )


def test_view_none_returns_all_columns():
    t = _obs_table()
    result = view(t, None)
    assert result.colnames == t.colnames
    assert result is not t  # a copy


def test_view_selects_named_subset():
    t = _obs_table()
    result = view(t, "OBSLOG")
    # every returned column is in the OBSLOG set and present in the table
    assert set(result.colnames).issubset(set(SUMMARY_COLUMNS["OBSLOG"]))
    assert "MAIN_ID" in result.colnames


def test_view_silently_drops_absent_columns():
    t = _obs_table()  # lacks most MEDIUM columns
    result = view(t, "MEDIUM")
    assert "MAIN_ID" in result.colnames
    assert "GAIA_TEFF" not in result.colnames  # absent in this table, dropped


def test_view_unknown_summary_raises():
    with pytest.raises(KeyError):
        view(_obs_table(), "NOPE")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: FAIL with `ImportError: cannot import name 'SUMMARY_COLUMNS'`.

- [ ] **Step 3: Write minimal implementation**

After `usable_mask` in `sphere_database.py`, add. (These lists are the existing `_keys_for_*` sets with the `{FILTER}`/`{READY}` placeholders resolved to the literals `"FILTER"` and `"HCI_READY"`.)

```python
_HEAD = ["MAIN_ID"]
_TAIL = ["NIGHT_START", "FILTER", "WAFFLE_MODE", "HCI_READY", "DEROTATOR_MODE", "PRIMARY_SCIENCE"]

SUMMARY_COLUMNS: dict = {
    "NORMAL": _HEAD
    + ["ID_GAIA_DR3", "ID_HIP", "RA", "DEC", "OTYPE", "SP_TYPE", "FLUX_H", "STARS_IN_CONE"]
    + _TAIL
    + ["TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM", "OBS_PROG_ID", "TOTAL_FILE_SIZE_MB"],
    "OBSLOG": _HEAD
    + _TAIL
    + ["DIT", "NDIT", "NCUBES", "TOTAL_EXPTIME_SCI", "TOTAL_EXPTIME_FLUX",
       "ROTATION", "MEAN_FWHM", "MEAN_TAU", "OBS_PROG_ID"],
    "SHORT": _HEAD
    + _TAIL
    + ["GAIA_TEFF", "MOCA_AID", "MOCA_BANYAN_PROB", "MOCA_AGE_MYR",
       "TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM", "OBS_PROG_ID"],
    "MEDIUM": _HEAD
    + _TAIL
    + ["GAIA_TEFF", "GAIA_LOGG", "GAIA_MH", "MOCA_AID", "MOCA_BANYAN_PROB",
       "MOCA_ASSOCIATION_NAME", "MOCA_ASSOCIATION_TYPE", "MOCA_AGE_MYR", "MOCA_AGE_MYR_UNC",
       "FLUX_H", "DIT", "NDIT", "TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM",
       "STDDEV_FWHM", "OBS_PROG_ID"],
}


def view(table: Table, summary: str | None = None) -> Table:
    """Return a copy of ``table`` reduced to a named summary column set.

    Parameters
    ----------
    table : astropy.table.Table
        Any observation table (filtered or not).
    summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'} or None
        Named column set. Columns not present in ``table`` are silently
        dropped. ``None`` returns all columns.

    Returns
    -------
    astropy.table.Table
        A copy with the selected columns.

    Raises
    ------
    KeyError
        If ``summary`` is not a known name.
    """
    if summary is None:
        return table.copy()
    columns = [c for c in SUMMARY_COLUMNS[summary] if c in table.colnames]
    return table[columns].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/spherical/database/sphere_database.py tests/test_filter.py
git commit -m "feat(database): add SUMMARY_COLUMNS and standalone view()"
```

---

## Task 3: Criterion helpers + `filter()` core + `columns` property

**Files:**
- Modify: `src/spherical/database/sphere_database.py` (add `_isin`, `_criterion_mask` module-level after `view`; add `filter` method and `columns` property to `SphereDatabase`)
- Modify: `CHANGELOG.md`
- Test: `tests/test_filter.py`

**Interfaces:**
- Consumes: `usable_mask` (Task 1).
- Produces:
  - `_isin(column, values) -> np.ndarray` — membership, string-safe (stringifies bytes/unicode columns before `np.isin`).
  - `_criterion_mask(column, cond) -> np.ndarray` — scalar→equality, list/set/ndarray→membership, `('not in', seq)`→exclusion.
  - `SphereDatabase.columns -> list[str]` (property) == `self.table_of_observations.colnames`.
  - `SphereDatabase.filter(*masks, usable_only=False, target_list=None, **criteria) -> Table`. In this task `target_list` is accepted but not yet used (Task 4 adds resolution); default `None`. Returns a copied `Table`; missing values excluded per criterion; unknown criteria column raises `KeyError` with a difflib suggestion.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_filter.py`:

```python
from spherical.database.sphere_database import SphereDatabase


def _filter_table():
    return Table(
        {
            "MAIN_ID": ["a", "b", "c", "d"],
            "MEAN_FWHM": [0.8, 1.5, 0.9, 0.7],
            "FLUX_J": [6.0, 7.0, 9.0, 5.0],
            "DISTANCE": [10.0, 50.0, 20.0, 30.0],
            "MOCA_ASSOCIATION_NAME": ma.MaskedArray(
                [b"bpmg", b"other", b"bpmg", b"zzz"], mask=[False, False, False, True]
            ),
            "OBS_PROG_ID": [b"095.C-0298", b"111.24QL.001", b"110.240D.001", b"095.C-0298"],
            "MOCA_AGE_MYR": ma.MaskedArray([20.0, 30.0, 99.0, 10.0], mask=[False, True, False, False]),
        }
    )


def _lightweight_db(table):
    db = SphereDatabase.__new__(SphereDatabase)  # bypass heavy __init__
    db.table_of_observations = table
    return db


def test_filter_equality_criterion():
    db = _lightweight_db(_filter_table())
    result = db.filter(MOCA_ASSOCIATION_NAME="bpmg")
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"a", "c"}


def test_filter_membership_list():
    db = _lightweight_db(_filter_table())
    result = db.filter(OBS_PROG_ID=["110.240D.001", "111.24QL.001"])
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b", "c"}


def test_filter_not_in_exclusion():
    db = _lightweight_db(_filter_table())
    result = db.filter(OBS_PROG_ID=("not in", ["095.C-0298"]))
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b", "c"}


def test_filter_mask_and_criteria_compose():
    db = _lightweight_db(_filter_table())
    result = db.filter(lambda t: t["MEAN_FWHM"] < 1.0, MOCA_ASSOCIATION_NAME="bpmg")
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"a", "c"}


def test_filter_cross_column_mask():
    db = _lightweight_db(_filter_table())
    # absolute J mag = FLUX_J - 5*log10(DISTANCE/10); keep intrinsically bright.
    # a: 6-0=6.0, b: 7-3.49=3.51, c: 9-1.51=7.49, d: 5-2.39=2.61  -> < 5 keeps {b, d}
    result = db.filter(lambda t: t["FLUX_J"] - 5 * np.log10(t["DISTANCE"] / 10) < 5)
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b", "d"}


def test_filter_excludes_missing_for_criterion():
    db = _lightweight_db(_filter_table())
    # row 'd' has MOCA_ASSOCIATION_NAME masked -> excluded even from "not equal to bpmg"
    result = db.filter(MOCA_ASSOCIATION_NAME=("not in", ["bpmg"]))
    assert "d" not in set(np.asarray(result["MAIN_ID"]).astype(str))
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"b"}


def test_filter_excludes_missing_for_numeric_criterion():
    db = _lightweight_db(_filter_table())
    # row 'b' has MOCA_AGE_MYR masked -> excluded
    result = db.filter(MOCA_AGE_MYR=("not in", [999.0]))
    assert "b" not in set(np.asarray(result["MAIN_ID"]).astype(str))


def test_filter_unknown_column_suggests():
    db = _lightweight_db(_filter_table())
    with pytest.raises(KeyError, match="MEAN_FWHM"):
        db.filter(MEAN_FWH=0.8)


def test_filter_mask_wrong_length_raises():
    db = _lightweight_db(_filter_table())
    with pytest.raises(ValueError):
        db.filter(np.array([True, False]))  # len 2 != 4


def test_columns_property():
    db = _lightweight_db(_filter_table())
    assert db.columns == db.table_of_observations.colnames
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: FAIL — `AttributeError: 'SphereDatabase' object has no attribute 'filter'`.

- [ ] **Step 3: Write minimal implementation**

After `view()` in `sphere_database.py`, add the module-level helpers:

```python
def _isin(column, values) -> np.ndarray:
    """Membership test that is safe for bytes/unicode string columns.

    ``numpy.isin`` silently fails to match a ``|S`` (bytes) column against a
    list of ``str`` values, so string columns are stringified on both sides.
    """
    data = np.asarray(column)
    if data.dtype.kind in ("S", "U"):
        return np.isin(data.astype(str), np.asarray(list(values)).astype(str))
    return np.isin(data, list(values))


def _criterion_mask(column, cond) -> np.ndarray:
    """Boolean mask for one filter criterion on a single column.

    ``cond`` is a scalar (equality), a list/set/ndarray (membership), or a
    ``('not in', sequence)`` tuple (exclusion). Missing-value handling is done
    by the caller via the column mask, so masked positions may take an
    arbitrary value here.
    """
    if isinstance(cond, tuple):
        if len(cond) == 2 and cond[0] == "not in":
            values = cond[1]
            if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
                raise TypeError("('not in', ...) requires a sequence of values.")
            return ~_isin(column, values)
        raise ValueError("Tuple criteria must be of the form ('not in', sequence).")
    if isinstance(cond, (list, set, np.ndarray)):
        return _isin(column, cond)
    return np.asarray(column == cond)
```

Then add to the `SphereDatabase` class (place near `return_usable_only`):

```python
    @property
    def columns(self) -> List[str]:
        """Column names available for filtering."""
        return self.table_of_observations.colnames

    def filter(self, *masks, usable_only: bool = False, target_list=None, **criteria) -> Table:
        """Return observations matching all given criteria and masks.

        Parameters
        ----------
        *masks : numpy.ndarray or callable
            Boolean arrays, or callables ``f(table) -> bool array``, combined
            with logical AND. Use these for comparisons and cross-column
            arithmetic (e.g. absolute magnitude, colours).
        usable_only : bool, optional
            If True, restrict to high-contrast-usable observations
            (see :func:`usable_mask`).
        target_list : list of str, optional
            Restrict to these targets first (resolved by name; see
            :meth:`observations_from_name_SIMBAD`). ``None`` uses all rows.
        **criteria : scalar, list, or ('not in', sequence)
            Per-column tests: a scalar means equality, a list means membership,
            and ``('not in', seq)`` means exclusion. A row whose value for a
            criterion's column is missing is always excluded.

        Returns
        -------
        astropy.table.Table
            A copy of the matching rows (all columns). Empty if nothing matches.

        Raises
        ------
        KeyError
            If a criterion names a column not in the table.
        ValueError
            If a boolean-array mask has the wrong length.
        """
        import difflib

        table = self.table_of_observations

        mask = np.ones(len(table), dtype=bool)

        for column_name, cond in criteria.items():
            if column_name not in table.colnames:
                suggestion = difflib.get_close_matches(column_name, table.colnames, n=1)
                hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                raise KeyError(f"{column_name!r} is not a column.{hint}")
            column = table[column_name]
            present = ~np.ma.getmaskarray(column)
            mask &= present & _criterion_mask(column, cond)

        for spec in masks:
            m = spec(table) if callable(spec) else np.asarray(spec)
            if len(m) != len(table):
                raise ValueError(f"Mask length {len(m)} does not match table length {len(table)}.")
            mask &= np.ma.filled(np.ma.asarray(m), False)

        if usable_only:
            mask &= usable_mask(table)

        return table[np.ma.filled(mask, False)].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: PASS (all tests, including the missing-value ones).

- [ ] **Step 5: Update CHANGELOG**

Add under `## [Unreleased]` → `### ✨ Added` in `CHANGELOG.md`:

```markdown
- `SphereDatabase.filter()` for composable, validated observation-table filtering
  (equality/membership kwargs + numpy mask escape hatch), plus `view()` and a
  `columns` property. Missing values are excluded per criterion.
```

- [ ] **Step 6: Commit**

```bash
git add src/spherical/database/sphere_database.py tests/test_filter.py CHANGELOG.md
git commit -m "feat(database): add SphereDatabase.filter() with validated criteria and mask escape hatch"
```

---

## Task 4: `filter()` target_list resolution

**Files:**
- Modify: `src/spherical/database/sphere_database.py` (`filter` method — add the `target_list` branch)
- Test: `tests/test_filter.py`

**Interfaces:**
- Consumes: `SphereDatabase.observations_from_name_SIMBAD` (existing) and `filter()` core (Task 3).
- Produces: `filter(..., target_list=[...])` restricts to the resolved targets before applying masks/criteria. If no name resolves, returns an empty table.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_filter.py`:

```python
def _named_db():
    table = Table(
        {
            "MAIN_ID": ["* bet Pic", "* bet Pic", "* alf Cen", "HD 1"],
            "MEAN_FWHM": [0.8, 1.5, 0.9, 0.7],
        }
    )
    db = SphereDatabase.__new__(SphereDatabase)
    db.table_of_observations = table
    db._normalized_id_lookup = {"betapic": [0, 1], "alfcen": [2], "hd1": [3]}
    return db


def test_filter_target_list_restricts():
    db = _named_db()
    result = db.filter(target_list=["beta pic"])
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"* bet Pic"}
    assert len(result) == 2


def test_filter_target_list_with_criteria():
    db = _named_db()
    result = db.filter(target_list=["beta pic"], MEAN_FWHM=0.8)
    assert len(result) == 1


def test_filter_target_list_unresolved_returns_empty():
    from unittest.mock import patch

    db = _named_db()
    with patch("spherical.database.sphere_database.Simbad.query_object", return_value=None):
        result = db.filter(target_list=["does-not-exist"])
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_filter.py::test_filter_target_list_restricts -v`
Expected: FAIL — `target_list` currently ignored, so all 4 rows return (assert on 2 fails).

- [ ] **Step 3: Write minimal implementation**

In `filter()`, replace the line `table = self.table_of_observations` with the resolving version:

```python
        table = self.table_of_observations
        if target_list is not None:
            resolved = self.observations_from_name_SIMBAD(target_list)
            if resolved is None:
                return table[np.zeros(len(table), dtype=bool)].copy()
            resolved_ids = np.unique(np.asarray(resolved["MAIN_ID"]))
            table = table[np.isin(np.asarray(table["MAIN_ID"]), resolved_ids)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: PASS (all, including the three new target_list tests).

- [ ] **Step 5: Commit**

```bash
git add src/spherical/database/sphere_database.py tests/test_filter.py
git commit -m "feat(database): support target_list resolution in filter()"
```

---

## Task 5: Consolidate `usable_only`, remove `FAILED_SEQ`, repoint presentation to `view()`

**Files:**
- Modify: `src/spherical/database/sphere_database.py` (`__init__`, `_mask_not_usable_observations`, `return_usable_only`, `show_in_browser`, `observations_from_name_SIMBAD`, `get_observation_SIMBAD`)
- Modify: `tests/test_observations_from_name.py`
- Modify: `CHANGELOG.md`
- Test: existing `tests/test_filter.py`, `tests/test_observations_from_name.py`, `tests/test_database.py`

**Interfaces:**
- Consumes: `usable_mask` (Task 1), `view` (Task 2).
- Produces: no new public surface; `usable_only=` everywhere now routes through `usable_mask()`, summaries through `view()`. `_ready_flag`, `_mask_not_usable_observations`, `_not_usable_observations_mask`, `_keys_for_*`, and the local `_keys()` helper are gone.

- [ ] **Step 1: Update the name-resolution test fixture (remove obsolete attribute)**

In `tests/test_observations_from_name.py`, delete the line in `_make_lightweight_db`:

```python
    db._not_usable_observations_mask = np.zeros(len(tobs), bool)
```

(The tests call `observations_from_name_SIMBAD` without `usable_only`, so this attribute is no longer read.)

- [ ] **Step 2: Remove the `_ready_flag`, precomputed mask, and `_keys_for_*` block from `__init__`**

In `SphereDatabase.__init__`, delete these lines (currently ~138-141):

```python
        # ---------- 5) flag column name (HCI_READY ↔ ~FAILED_SEQ) ---------------
        self._ready_flag = "HCI_READY" if "HCI_READY" in self.table_of_observations.colnames else "FAILED_SEQ"

        self._not_usable_observations_mask = self._mask_not_usable_observations(5.0)
```

And delete the entire "build lists of summary keys" block (currently ~143-229): the `def _keys(base)` helper, `base_head`, `base_tail`, and all four `self._keys_for_*` assignments. These are replaced by module-level `SUMMARY_COLUMNS` (Task 2).

- [ ] **Step 3: Delete `_mask_not_usable_observations` and rewrite `return_usable_only`**

Delete the whole `_mask_not_usable_observations` method (currently ~255-281). Rewrite `return_usable_only` to use the module function:

```python
    def return_usable_only(self) -> Table:
        """Return a copy of the table with only usable observations.

        Returns
        -------
        astropy.table.Table
            Observations flagged usable by :func:`usable_mask`.
        """
        return self.table_of_observations[usable_mask(self.table_of_observations)].copy()
```

- [ ] **Step 4: Rewrite `show_in_browser` to use `usable_mask` + `view`**

Replace the body of `show_in_browser`:

```python
    def show_in_browser(self, summary: Optional[str] = None, usable_only: bool = False) -> None:
        """Display the observation table in a web browser using JSViewer.

        Parameters
        ----------
        summary : {'NORMAL', 'SHORT', 'MEDIUM', 'OBSLOG'}, optional
            Named column set; ``None`` shows all columns.
        usable_only : bool, optional
            If True, restrict to usable observations (see :func:`usable_mask`).
        """
        table = self.table_of_observations
        if usable_only:
            table = table[usable_mask(table)]
        view(table, summary).show_in_browser(jsviewer=True)
```

- [ ] **Step 5: Repoint the SIMBAD-lookup summary/usable branches**

In `observations_from_name_SIMBAD`, replace the usable branch:

```python
        if usable_only:
            table_of_observations = self.table_of_observations[usable_mask(self.table_of_observations)].copy()
        else:
            table_of_observations = self.table_of_observations.copy()
```

and replace the trailing summary `if/elif` chain (currently ~563-572) with:

```python
        return view(matching_observations, summary)
```

In `get_observation_SIMBAD`, the `usable_only` argument is forwarded to `observations_from_name_SIMBAD`, so no change is needed there beyond confirming it still passes through.

- [ ] **Step 6: Run the full database test suite**

Run: `pixi run -e dev pytest tests/test_filter.py tests/test_observations_from_name.py tests/test_database.py -v`
Expected: PASS. (`test_database.py` does not reference any removed internal — verified — so no edits are expected there; if a run surprises you, repoint to `view()` / `usable_mask()` / `return_usable_only()`.)

- [ ] **Step 7: Lint**

Run: `pixi run -e dev ruff check src/spherical/database/sphere_database.py`
Expected: no new errors. (Remove the now-unused `_ready_flag` references and any newly-unused imports.)

- [ ] **Step 8: Update CHANGELOG**

Add under `## [Unreleased]`:

Use the existing `### Changed` and `### Removed` subsections under `## [Unreleased]` (do not create new ones):

```markdown
### Changed
- `usable_only` filtering consolidated into a single `usable_mask()`; the
  precomputed mask and `_mask_not_usable_observations()` are removed (behaviour
  unchanged). Summary column sets moved to a module-level `SUMMARY_COLUMNS`
  used by `view()`.

### Removed
- Deprecated `FAILED_SEQ` / `_ready_flag` readiness path (superseded by
  `HCI_READY`).
```

- [ ] **Step 9: Commit**

```bash
git add src/spherical/database/sphere_database.py tests/test_observations_from_name.py tests/test_database.py CHANGELOG.md
git commit -m "refactor(database): consolidate usable_only into usable_mask, drop FAILED_SEQ, route summaries through view()"
```

---

## Task 6: `ROTATION` sentinel `-10000` → `np.nan`

**Files:**
- Modify: `src/spherical/database/observation_table.py:154,164`
- Test: `tests/test_filter.py`

**Interfaces:**
- Produces: `ROTATION` is `np.nan` (not `-10000`) when angle computation is skipped (polarimetry) or fails, so it round-trips to a masked column and the filter's presence logic excludes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_filter.py`:

```python
def test_rotation_missing_is_nan_and_excluded_by_filter():
    # A missing ROTATION (NaN) round-trips to a masked column; a ROTATION
    # criterion must exclude it rather than treat -10000 as a real value.
    t = Table(
        {
            "MAIN_ID": ["a", "b"],
            "ROTATION": ma.MaskedArray([25.0, np.nan], mask=[False, True]),
        }
    )
    db = _lightweight_db(t)
    result = db.filter(ROTATION=("not in", [-10000.0]))
    assert set(np.asarray(result["MAIN_ID"]).astype(str)) == {"a"}
```

Also add a direct check of the build helper. Append:

```python
def test_compute_rotation_returns_nan_for_polarimetry():
    from spherical.database.observation_table import compute_derotation_info

    result = compute_derotation_info(observation_files=Table(), polarimetry=True)
    assert np.isnan(result["ROTATION"])
    assert result["DEROTATOR_FLAG"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_filter.py::test_compute_rotation_returns_nan_for_polarimetry -v`
Expected: FAIL — returns `-10000`, not NaN.

- [ ] **Step 3: Write minimal implementation**

In `src/spherical/database/observation_table.py`, change both return sites (lines ~154 and ~164):

```python
        return {"ROTATION": np.nan, "DEROTATOR_FLAG": True}
```

(Confirm `import numpy as np` is present at the top of the file; it is used elsewhere in this module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Update CHANGELOG**

Add under `## [Unreleased]` → the existing `### Fixed`:

```markdown
- `ROTATION` now stores `np.nan` (not the `-10000` sentinel) when derotation is
  not applicable/failed, so it round-trips to a masked column and is treated as
  missing by filtering. Takes effect on the next database rebuild.
```

- [ ] **Step 6: Commit**

```bash
git add src/spherical/database/observation_table.py tests/test_filter.py CHANGELOG.md
git commit -m "fix(database): use np.nan instead of -10000 sentinel for missing ROTATION"
```

---

## Task 7: Update the IFS reduction template

**Files:**
- Modify: `examples/ifs_reduction_template.py:114-127`

**Interfaces:**
- Consumes: `filter()` (Tasks 3-4).

- [ ] **Step 1: Replace the target-resolution + manual-mask block**

In `examples/ifs_reduction_template.py`, replace the current block (currently lines ~114-127):

```python
observation_table = database.target_list_to_observation_table(target_list)
# Apply filters to select observations
observation_table_mask = np.logical_and.reduce([
    observation_table['TOTAL_EXPTIME_SCI'] > 30,
    observation_table['DEROTATOR_MODE'] == 'PUPIL',
    observation_table['HCI_READY'] == True,]
)
# Another useful keyword is 'OBS_PROG_ID', the program ID of the survey you want to reduce.

observation_table = observation_table[observation_table_mask]
```

with:

```python
# Select observations for the requested targets. `usable_only` requires
# HCI-ready, pupil-stabilized data above a minimum exposure; add any extra
# thresholds as masks and equality/membership criteria as keywords.
# e.g. OBS_PROG_ID=('not in', ['095.C-0298']) to exclude a program.
observation_table = database.filter(
    lambda t: t['TOTAL_EXPTIME_SCI'] > 30,
    target_list=target_list,
    usable_only=True,
)
```

- [ ] **Step 2: Verify the script imports and builds the table**

Run (requires the pipeline extra + local data; if unavailable, at minimum syntax-check):
`pixi run -e dev python -c "import ast; ast.parse(open('examples/ifs_reduction_template.py').read())"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add examples/ifs_reduction_template.py
git commit -m "docs(examples): use database.filter(target_list=...) in IFS reduction template"
```

---

## Task 8: Update the exploration notebook

**Files:**
- Modify: `explore_database.ipynb` (the filtering cells + a short markdown note)

**Interfaces:**
- Consumes: `filter()`, `view()` (Tasks 2-4).

- [ ] **Step 1: Replace the filtering example cell**

Replace the `np.logical_and.reduce([...])` filtering cell (the "Example of filtering for observations" section) with:

```python
obs = database.filter(
    lambda t: (t['MEAN_FWHM'] < 1.2) & (t['FLUX_J'] < 8) & (t['DISTANCE'] < 100),
    usable_only=True,
    MOCA_ASSOCIATION_NAME='beta Pictoris moving group',
    OBS_PROG_ID=('not in', ['095.C-0298']),  # exclude a program; or a bare list to include only these
)
print(f"Number of total observations: {len(database.table_of_observations)}")
print(f"Number of matching observations: {len(obs)}")
view(obs, 'OBSLOG').show_in_browser(jsviewer=True)
```

Ensure the import cell includes `view`:

```python
from spherical.database.sphere_database import SphereDatabase, view
```

- [ ] **Step 2: Add a markdown cell explaining `HCI_READY` vs `usable_only`**

Insert a markdown cell before the filtering example:

```markdown
**`HCI_READY` vs `usable_only`.** `HCI_READY` is a per-sequence completeness
flag: does this observation have the centering and flux-calibration frames,
consistent DITs, and valid derotation data it needs? `usable_only=True` is
stricter — it additionally requires the sequence to be **pupil-stabilized**
(`DEROTATOR_MODE != 'FIELD'`, i.e. usable for ADI) and above a **minimum total
exposure time**. A field-stabilized sequence can be `HCI_READY` yet not
`usable_only`. Use `usable_only=True` as the default high-contrast selection;
filter on `HCI_READY` directly only if you specifically want the looser cut.

**Filtering with `database.filter(...)`.** Pass numeric/cross-column conditions
as `lambda t: ...` masks and equality/membership as keywords (a scalar means
`==`, a list means "in", and `('not in', [...])` excludes). A row missing a
value for a criterion's column is always excluded. `database.columns` lists the
available column names.
```

- [ ] **Step 3: Restart-and-run the notebook to verify**

Run the notebook top-to-bottom (needs the database tables on disk). Confirm the filtering cell executes, the counts print, and the browser view opens. If the tables are not available locally, note this as manual verification pending data.

- [ ] **Step 4: Commit**

```bash
git add explore_database.ipynb
git commit -m "docs(notebook): use database.filter()/view() and explain HCI_READY vs usable_only"
```

---

## Self-Review Notes

- **Spec coverage:** filter syntax (Task 3), return type = Table (Task 3), `view()`/`SUMMARY_COLUMNS` (Task 2), missing-value exclusion (Tasks 3, 6 tests), `target_list` fold-in (Task 4), `usable_only` consolidation + `HCI_READY` distinction (Tasks 1, 5, 8), `FAILED_SEQ` removal (Task 5), `ROTATION` source fix (Task 6), consumer updates (Tasks 7-8), error handling (Task 3 tests), tests (throughout). `target_list_to_observation_table` intentionally kept (spec: later-deprecation candidate) — not removed here.
- **Type consistency:** `filter(*masks, usable_only=False, target_list=None, **criteria) -> Table`, `view(table, summary=None) -> Table`, `usable_mask(table) -> np.ndarray`, `_criterion_mask(column, cond) -> np.ndarray`, `_isin(column, values) -> np.ndarray`, `columns -> list[str]` are used consistently across tasks.
- **Placeholders:** none — all steps carry real code/commands. Verified facts baked in: the ROTATION helper is `compute_derotation_info` (Task 6), `test_database.py` references no removed internals (Task 5), and the CHANGELOG `[Unreleased]` subsections are `### ✨ Added` / `### Changed` / `### Removed` / `### Fixed`.

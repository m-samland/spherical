# Contributing

This page explains which contributions fit spherical, what a pull
request must contain, and how to set up an environment, run the tests and build these
pages. Contributions are welcome! For questions, open an
[issue](https://github.com/m-samland/spherical/issues).

## Before you start

Open an issue before you write code for anything larger than a small bug fix or a
documentation correction. Describe the problem and the change you have in mind, and
wait for agreement on the approach. A pull request that arrives without that
discussion may be closed if it does not fit the package.

## Scope

spherical has two halves, a database of SPHERE archive observations and an IFS and
IRDIS reduction that orchestrates charis and TRAP. Contributions that make either half
more correct or cover more of the archive are the most useful.

These contributions are welcome.

- Bug fixes with a minimal example that reproduces the problem.
- Support for SPHERE observing modes, header conventions or archive edge cases that
  the database handles wrongly or misses.
- New driver scripts in `examples/` for a target, survey or observing mode.
- New tests, in particular for code paths that are not yet covered.
- Corrections and additions to these pages.
- Performance improvements, together with a measurement before and after.

These contributions will not be accepted.

- Command-line options for the reduction. The reduction is configured through the
  Python dataclasses in `spherical.pipeline.pipeline_config`, and new analyses are new
  scripts in `examples/`.
- Changes to the internals of charis or TRAP. Open those in
  [charis-dep](https://github.com/PrincetonUniversity/charis-dep) or
  [TRAP](https://github.com/m-samland/trap) instead.
- New dependencies for the database half. `import spherical` must work without the
  `pipeline` extra, so heavy imports belong in `spherical.pipeline`.
- Large refactors, renames or reformatting without a prior issue.
- Unrelated changes bundled into one pull request. Open one pull request per change.

You may use AI tools to write code or text. You must have run the code yourself, and
you must be able to explain every change in the pull request. Pull requests whose
author cannot answer questions about the code will be closed.

## Checklist for a pull request

1. Work on a branch off `develop` or in a fork, open the pull request against
   `develop`, and link the issue it addresses.
2. Add or update a test for every change in behavior. Tests go in `tests/pipeline/` or
   `tests/database/` by subject. Mark a test that reaches a live service, including
   through a fixture, with `pytestmark = pytest.mark.remote_data`.
3. Run the test suite and make sure it passes, as described under [Tests](#tests). If
   you changed the database, also run the `remote_data` tests.
4. Run the linter, as described under [Lint](#lint), and fix every error in the files
   you touched.
5. If you changed `spherical.database`, check that `import spherical` works without
   the `pipeline` extra. `tests/database/test_base_install_imports.py` checks this for
   the imports it can see.
6. Update these pages if the change affects how the package is used, following the
   [writing guide](writing-docs.md).
7. If you changed a reduction step, say in the pull request which sequence you reduced
   end to end and which products you checked. CI cannot run a reduction, so this is
   the only evidence that the step still works on real data.
8. Fill in the pull request template.

Small pull requests that do one thing are reviewed faster. The review checks that the
change fits the scope above, that tests cover it, and that the code matches the style
of the file it edits.

## Development environment

::::{tab-set}
:sync-group: installer

:::{tab-item} pip
:sync: pip

```bash
git clone https://github.com/m-samland/spherical.git
cd spherical
pip install -e ".[pipeline,notebook,test]"
```
:::

:::{tab-item} pixi
:sync: pixi

```bash
git clone https://github.com/m-samland/spherical.git
cd spherical
pixi install -e dev
pixi shell -e dev
```

The `dev` environment installs charis and TRAP in editable mode from
`../charis-dep` and `../trap`, so clone both next to `spherical` first. To work on
spherical alone, use `pixi install -e dev-git`, which takes charis from GitHub and
TRAP from PyPI.
:::
::::

## Tests

A plain `pytest` runs the offline tests of the database (`tests/database`), the
pipeline (`tests/pipeline`) and the documentation (`tests/docs`).

```bash
pytest
# or, with pixi, including the tests that need charis and TRAP
pixi run -e dev test
```

Tests that need charis are skipped unless the `pipeline` extra is installed, and
tests that need TRAP skip without it. `pixi run -e test test` runs the suite without
charis and TRAP. CI installs `.[test]`, which includes TRAP but not charis, and runs
the database, pipeline and docs tests on every push.

Two sets are opt-in.

- `pytest tests/database -m remote_data` (or `pixi run -e dev test-network`) queries
  the live ESO archive and takes about 3 minutes.
- `pytest tests/regression -m regression` (or `pixi run -e dev test-regression`)
  compares a 51 Eri reduction with frozen astrometry baselines and checks the
  result against a GRAVITY measurement kept as an internal reference. It needs the
  full pipeline and the reduced data on disk; the
  [benchmark notes](https://github.com/m-samland/spherical/blob/develop/tests/regression/data/51eri_astrometry_benchmark.md)
  describe how to produce them.

The reductions are too heavy for CI. To test one end to end, install the full
pipeline, pick a small sequence, run `examples/ifs_reduction_template.py` or
`examples/irdis_reduction_template.py`, and inspect the logs and products.

## Lint

```bash
ruff check src/ tests/
# or, with pixi, which also checks the documentation tools
pixi run -e dev lint
```

## Documentation

Build the site with `pixi run -e docs docs-clean && pixi run -e docs docs`. The
[writing guide](writing-docs.md) covers voice, page shape and the checks a page
must pass.

```{toctree}
:maxdepth: 1

writing-docs
```

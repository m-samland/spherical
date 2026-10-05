# Contributing

Set up a development environment, run the tests and build these pages.
Contributions are welcome. For questions, open an
[issue](https://github.com/m-samland/spherical/issues).

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

Tests that need charis are skipped unless the `pipeline` extra is installed.
`pixi run -e test test` runs the suite without charis and TRAP, as CI does for the
database half on every push.

Two sets are opt-in.

- `pytest tests/database -m remote_data` queries the live ESO archive and takes
  about 20 minutes.
- `pytest tests/regression -m regression` (or `pixi run -e dev test-regression`)
  compares a 51 Eri reduction with frozen astrometry baselines and checks the
  result against the published GRAVITY position. It needs the full pipeline and the
  reduced data on disk; the
  [benchmark notes](https://github.com/m-samland/spherical/blob/develop/tests/regression/data/51eri_astrometry_benchmark.md)
  describe how to produce them.

The reductions are too heavy for CI. To test one end to end, install the full
pipeline, pick a small sequence, run `examples/ifs_reduction_template.py` or
`examples/irdis_reduction_template.py`, and inspect the logs and products.

## Lint

```bash
ruff check src/ tests/
# or
pixi run -e dev lint
```

## Pull requests

Work on a feature branch or a fork and open the pull request against `develop`,
using the issue and pull request templates. Run the tests and the linter first.

## Documentation

Build the site with `pixi run -e docs docs-clean && pixi run -e docs docs`. The
[writing guide](writing-docs.md) covers voice, page shape and the checks a page
must pass.

```{toctree}
:maxdepth: 1

writing-docs
```

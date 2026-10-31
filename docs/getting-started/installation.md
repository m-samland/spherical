# Installation

After this page you have spherical installed, either to explore the observation
database or to run the full reduction pipeline. You need Python 3.11 or newer on
Linux or macOS.

The database half needs only common scientific Python packages. The full pipeline
adds {term}`charis` and {term}`TRAP` for the IFS and IRDIS reductions.

::::::{tab-set}
:sync-group: installer

:::::{tab-item} pip
:sync: pip

Create a dedicated environment first, for example with mamba.

```bash
mamba create -n spherical_env python=3.13
mamba activate spherical_env
```

::::{tab-set}
:sync-group: install-scope

:::{tab-item} Database only
:sync: database

```bash
pip install git+https://github.com/m-samland/spherical.git
```
:::

:::{tab-item} Full pipeline
:sync: pipeline

```bash
pip install "spherical[pipeline] @ git+https://github.com/m-samland/spherical.git"
```
:::
::::
:::::

:::::{tab-item} pixi
:sync: pixi

[Pixi](https://pixi.prefix.dev/) creates the environment for you from the repository.

```bash
git clone https://github.com/m-samland/spherical.git
cd spherical
```

::::{tab-set}
:sync-group: install-scope

:::{tab-item} Database only
:sync: database

```bash
pixi install
```
:::

:::{tab-item} Full pipeline
:sync: pipeline

```bash
pixi install -e pipeline
```
:::
::::
:::::
::::::

## Check the installation

```bash
python -c "import spherical; print(spherical.__version__)"
```

This prints the installed version, for example `3.1.1.dev122+gde037ad14` for a
development install. For the full pipeline, also check that charis and TRAP import.

```bash
python -c "import charis, trap"
```

## Calibration data for the IFS reduction

charis needs static calibration files for SPHERE (bad-pixel masks, flats, PSFlet
templates and wavelength solutions), about 86 MB. It downloads them from Zenodo the
first time it reduces IFS data and keeps them in `~/.cache/charis/calibrations`, or in
`$XDG_CACHE_HOME/charis/calibrations` if that variable is set. Later runs read the
stored copy without network access.

To keep the files somewhere else, set `CHARIS_CALIBRATION_DIR`, for example in your
shell profile. `charis-calibrations` downloads them before the first reduction and
prints the directory in use.

```bash
export CHARIS_CALIBRATION_DIR=~/data/charis-calibrations
charis-calibrations fetch SPHERE
charis-calibrations path
```

Each charis release uses one fixed version of the data and stores it in its own
subfolder, so a charis update can add a new version next to the old one. Old versions
are not deleted automatically. Remove them by hand when no installed charis uses them.
On a cluster, download the data before the jobs start
([Run on a server or cluster](../how-to/server.md#download-the-charis-calibration-data-in-advance)).

## Troubleshooting

`pip install spherical` installs a different package
: The name `spherical` on PyPI belongs to an unrelated package. Install from GitHub
  as shown above.

The pixi `dev` environment fails to resolve
: It installs charis and TRAP in editable mode from `../charis-dep` and `../trap`,
  so both repositories must be cloned next to `spherical`. Use
  `pixi install -e dev-git` instead to get charis from GitHub and TRAP from PyPI.

Windows
: spherical is not supported on Windows.

To set up a development environment, see [Contributing](../contributing/index.md).

Next: [Get the database](database.md)

# Next steps

Where to go once a query works.

## Reduce data

The reductions run from a Python script. Start from the template for your
instrument, change the target list and the directories, and run it.

- [`examples/ifs_reduction_template.py`](https://github.com/m-samland/spherical/blob/develop/examples/ifs_reduction_template.py)
- [`examples/irdis_reduction_template.py`](https://github.com/m-samland/spherical/blob/develop/examples/irdis_reduction_template.py)

Every setting in the templates is described in [Configuration](../reference/configuration.md)
and [TRAP settings](../reference/trap-settings.md).

## Look things up

- [Pipeline steps](../reference/steps.md) lists what a reduction does, in order.
- [Conventions](../reference/conventions.md) explains the products: centre, rotation
  angle, units and file layout.
- [Command-line tools](../reference/cli.md) covers the database commands and the
  tools that summarise finished reductions.
- [Python API](../reference/api/index.md) documents every module.

## Explore interactively

The notebook
[`examples/explore_database.ipynb`](https://github.com/m-samland/spherical/blob/develop/examples/explore_database.ipynb)
shows more ways to search and plot the database.

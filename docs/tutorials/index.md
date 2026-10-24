# Tutorials

Each tutorial takes you through one complete task with real data and real output. After
working through them you can search the database for your own targets and know what a
finished reduction looks like.

[Exploring the database](exploring_the_database.ipynb) shows how to load the observation
tables, filter them by quality and target, and read the Gaia and MOCA columns. It runs in
a minute on a laptop once the database is installed.

[51 Eri b with IRDIS](51eri_irdis.ipynb) and [51 Eri b with IFS](51eri_ifs.ipynb) reduce
the same sequence of 51 Eri with each instrument, check the intermediate products and
measure the position and contrast of the planet 51 Eri b. They start from the reduction
templates and show how to search around a known companion.

## How these tutorials are made

Every page starts with a line that names the spherical version, the date and the machine
it was run on. The outputs on the pages come from that run and are not edited. A
reduction takes far longer than rendering a page, so a script in `docs/tutorials/runs/` runs it once, and the
reduction tutorials show that script and read the products it wrote.

```{toctree}
:maxdepth: 1

exploring_the_database
51eri_irdis
51eri_ifs
```

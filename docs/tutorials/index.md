# Tutorials

Each tutorial takes you through one complete task with real data and real output. After
working through them you can search the database for your own targets and know what a
finished reduction looks like.

[Exploring the database](exploring_the_database.ipynb) shows how to load the observation
tables, filter them by quality and target, and read the Gaia and MOCA columns. It runs in
a minute on a laptop once the database is installed.

## How these tutorials are made

Every page starts with a line that names the spherical version, the date and the machine
it was run on. The outputs on the pages come from that run and are not edited. A
reduction takes hours, so the reduction tutorials are made in two parts. A script in
`docs/tutorials/runs/` reduces the data once, and the page reads the products it wrote.
The page shows that script and the template functions it uses, so you see the code that
produced the results.

```{toctree}
:maxdepth: 1

exploring_the_database
```

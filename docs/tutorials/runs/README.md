# Tutorial runs

The reduction tutorials in `docs/tutorials/` read products from
`$SPHERICAL_TUTORIAL_DIR` (default `~/data/sphere_tutorials`). The scripts here write
them. This runbook is for maintainers refreshing the tutorials. Rerun a reduction only
when the 51 Eri regression tests fail, a product format changes, or a rendered tutorial
warns that the template changed since its run. Otherwise re-render the notebooks.

Each run script uses its reduction template unchanged and adds the TRAP settings of the
51 Eri regression run (`tests/regression/run_51eri_*_reference.py`): the annulus around
51 Eri b and its position as a known companion. `tests/pipeline/test_templates.py`
checks both.

Never run `cleanup_pipeline_products` on the tutorial directory. The notebooks read
from it for as long as the tutorials exist.

## Layout

```text
$SPHERICAL_TUTORIAL_DIR/
├── database/                  v3.0.0 tables (spherical-sync-tables --dest ...)
├── data/                      raw data, shared by all runs
├── species/                   private copy of the species database (see below)
├── 51eri_irdis/               one folder per run label
│   ├── run_provenance.json    versions, machine, set_ncpu, source hash
│   └── reduction/             IRDIS/{calibration,observation,trap}/...
├── 51eri_ifs/ ...
├── logs/                      one log per step of run_all.sh, and steps.log
└── csv/                       measure_requirements.py output
```

## The species database

TRAP's template matching adds its model grids to the species database at every
detection, so each run writes to `species_database.hdf5`. Give the runs a private copy
of a complete database instead of the one other work uses, on a disk with room for it
(about 15 GB), not on a node's small `/tmp`. Compute nodes without internet access
cannot download filter profiles or the Vega spectrum, so start from a database that
already holds them and rewrite its paths:

```bash
SRC=/path/to/species                       # a complete database (filters, spectra, models)
DST=$SPHERICAL_TUTORIAL_DIR/species
mkdir -p $DST && cp $SRC/species_database.hdf5 $DST/
printf '[species]\ndatabase = %s\ndata_folder = %s\nvega_mag = 0.03\n' \
    "$DST/species_database.hdf5" "$SRC/data" > $DST/species_config.ini
```

## On a machine without pixi (server)

Use a conda or mamba environment with Python 3.11 to 3.13.

```bash
mamba activate spherical   # or the name of your environment
git fetch && git checkout <branch with the run scripts>
pip install -e ".[pipeline]"
python -c "import charis, trap, spherical"
export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
spherical-sync-tables --dest "$SPHERICAL_TUTORIAL_DIR/database" --instrument all
```

`run_all.sh` runs the IRDIS and then the IFS tutorial reduction, each followed by its
measurement. The run scripts refuse a run folder that is not empty, so move an earlier
`51eri_irdis/` or `51eri_ifs/` aside first. Start it inside `screen` or `tmux`, and run
it as a script rather than pasting its lines, since `set -e` would close the shell at
the first error.

```bash
screen -S tutorials
mamba activate spherical   # the shell inside screen may not inherit it
cd /path/to/spherical
export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
export SPECIES_DIR=$SPHERICAL_TUTORIAL_DIR/species
export NCPU=48
bash docs/tutorials/runs/run_all.sh
```

Detach with `Ctrl-a d` and reattach with `screen -r tutorials`. `logs/steps.log`
records the start, end and machine load of every step. On a shared node the load of
other jobs changes the timings, so read them as rough numbers.

`measure_requirements.py` uses only the standard library and must run where the data
are, since it measures folder sizes.

## On a machine with pixi

```bash
pixi run -e test spherical-sync-tables --dest ~/data/sphere_tutorials/database --instrument all
pixi run -e dev python docs/tutorials/runs/51eri_irdis.py --ncpu 8
pixi run -e dev python docs/tools/measure_requirements.py --instrument irdis --label 51eri_irdis \
    --target "*_51_Eri" --filter DB_K12 --date 2015-09-24
```

Commit nothing between the run and the render that changes the run scripts or the
template functions, and run from a clean checkout: the provenance records
`git describe`.

## Copy the products to the machine that renders

The notebooks need the products, the logs and the TRAP results, but not the raw data,
the per-frame extracted cubes or the large science cubes. What is left is about 2 GB for
IRDIS (flux cubes) and 1.5 GB for IFS.

```bash
SRV=server:/path/with/space/sphere_tutorials
DST=~/data/sphere_tutorials
for run in 51eri_irdis 51eri_ifs; do
  rsync -av --prune-empty-dirs \
    --exclude='optext/CORO/' --exclude='optext/CENTER/' --exclude='optext/FLUX/' \
    --exclude='coro_cube*.fits' --exclude='coro_ivar_cube*.fits' \
    --exclude='center_cube*.fits' --exclude='center_ivar_cube*.fits' \
    --exclude='polychrome*' \
    "$SRV/$run/" "$DST/$run/"
done
rsync -av "$SRV/csv/" "$DST/csv/"
rsync -av "$SRV/logs/" "$DST/server_logs/"
```

Then render with `pixi run -e dev docs-tutorials 51eri_irdis` and `... 51eri_ifs`. The
notebook stops with a list of missing files if the copy left out something it reads.

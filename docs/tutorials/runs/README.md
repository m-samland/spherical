# Tutorial runs

The reduction tutorials in `docs/tutorials/` read products from
`$SPHERICAL_TUTORIAL_DIR` (default `~/data/sphere_tutorials`). The scripts here write
them. This runbook is for maintainers refreshing the tutorials. Rerun a reduction only
when the 51 Eri regression tests fail, a product format changes, or a rendered tutorial
warns that the template changed since its run. Otherwise re-render the notebooks.

Never run `cleanup_pipeline_products` on the tutorial directory. The notebooks read
from it for as long as the tutorials exist.

## Layout

```text
$SPHERICAL_TUTORIAL_DIR/
├── database/                  v3.0.0 tables (spherical-sync-tables --dest ...)
├── data/                      raw data, shared by all runs
├── 51eri_irdis/               one folder per run label
│   ├── run_provenance.json    versions, machine, set_ncpu, source hash
│   └── reduction/             IRDIS/{calibration,observation,trap}/...
├── 51eri_irdis_ncpu4/         preprocessing-only timing run (--no-trap)
├── 51eri_irdis_annulus/       annulus timing run (hard links to 51eri_irdis)
└── 51eri_ifs/ ...
```

## On a machine with pixi (IRDIS, MacBook)

```bash
pixi run -e test spherical-sync-tables --dest ~/data/sphere_tutorials/database --instrument all
pixi run -e dev python docs/tutorials/runs/51eri_irdis.py --ncpu 8
pixi run -e dev python docs/tools/measure_requirements.py --instrument irdis --label 51eri_irdis \
    --target "*_51_Eri" --filter DB_K12 --date 2015-09-24
pixi run -e dev python docs/tutorials/runs/51eri_irdis.py --ncpu 4 --label 51eri_irdis_ncpu4 --no-trap
pixi run -e dev python docs/tools/measure_requirements.py --instrument irdis --label 51eri_irdis_ncpu4 \
    --target "*_51_Eri" --filter DB_K12 --date 2015-09-24
pixi run -e dev python docs/tutorials/runs/annulus_timing.py --instrument irdis --ncpu 8 --inner 31 --outer 43
pixi run -e dev docs-tutorials 51eri_irdis
```

Commit nothing between the run and the render that changes the template functions, and
run from a clean checkout: the provenance records `git describe`.

## On a machine without pixi (IFS, server)

Use a conda or mamba environment with Python 3.11 to 3.13.

```bash
git fetch && git checkout <branch with the run scripts>
pip install -e ".[pipeline]"
python -c "import charis, trap, spherical"
export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
spherical-sync-tables --dest "$SPHERICAL_TUTORIAL_DIR/database" --instrument all
```

`run_all.sh` runs every reduction one after the other: the IRDIS tutorial run, a
preprocessing-only IRDIS run at half the cores for the timing model, the IFS tutorial
run, and the two annulus timings, each followed by its measurement. It takes many hours,
so start it inside `screen` or `tmux`. Run it as a script and do not paste its lines into
a shell, since `set -e` would close the shell at the first error.

```bash
screen -S tutorials
cd /path/to/spherical
export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
export SPECIES_DIR=/path/to/species
export NCPU=48
bash docs/tutorials/runs/run_all.sh
```

Detach with `Ctrl-a d` and reattach with `screen -r tutorials`. Each step logs to
`$SPHERICAL_TUTORIAL_DIR/logs/`, and `logs/steps.log` records the start, end and machine
load of every step. Choose `NCPU` so that the run plus the other jobs on the node stay
below its core count. The load in `steps.log` shows whether the timings are usable.

`measure_requirements.py` uses only the standard library and must run where the data
are, since it measures folder sizes.

### Copy the products to the machine that renders

The notebook needs the products, the logs and the TRAP results, but not the raw data,
the per-frame extracted cubes or the large science cubes (about 1 GB instead of tens).

```bash
rsync -av --prune-empty-dirs \
    --include='*/' \
    --exclude='reduction/IFS/observation/*/*/*/optext/CORO/***' \
    --exclude='reduction/IFS/observation/*/*/*/optext/CENTER/***' \
    --exclude='reduction/IFS/observation/*/*/*/optext/FLUX/***' \
    --exclude='coro_cube*.fits' --exclude='center_cube*.fits' \
    --exclude='coro_ivar_cube*.fits' --exclude='center_ivar_cube*.fits' \
    --include='*' \
    server:"$SPHERICAL_TUTORIAL_DIR/51eri_ifs/" ~/data/sphere_tutorials/51eri_ifs/
rsync -av server:"$SPHERICAL_TUTORIAL_DIR/51eri_ifs_annulus/annulus_timing.json" \
    ~/data/sphere_tutorials/51eri_ifs_annulus/
rsync -av server:"$SPHERICAL_TUTORIAL_DIR/csv/" ./ifs_csv/
```

Then render with `pixi run -e dev docs-tutorials 51eri_ifs`. The notebook stops with a
list of missing files if the copy left out something it reads.

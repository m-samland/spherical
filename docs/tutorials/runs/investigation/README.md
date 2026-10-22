# Why 51 Eri b fails TRAP's shape check in the 2015 IRDIS tutorial run

The tutorial run of 2026-10-07 (`run_all.sh`, template settings) finds 51 Eri b at
37.3 px, PA 165.6 deg, SNR 5.97, but TRAP does not validate it. The free Gaussian fit to
the detection map gives `theta_deviation` -33 deg (limit 25) and `yx_fwhm_ratio` 0.62
(range 1.1 to 4.5). The 51 Eri regression reference validates the same planet with
`theta_deviation` 8.8 deg and ratio 1.24.

The inputs to TRAP are the same in both runs: `DEROT ANGLE` agrees to 1e-12 deg, the
centres to 0.004 px, and the flux PSF has the same shape (its peak differs by up to 2 %).
The TRAP settings differ:

| Setting | Regression | Tutorial |
|---|---|---|
| Search region | 31 to 43 px | 1 to 200 px |
| `temporal_components_fraction` | 0.20 | 0.15 |
| `yx_known_companion_position` | `[-35.95, -8.43]` | none |
| TRAP | 2.0.1 | 2.1.0 |

The shape is fitted in an 11 px box on the SNR map with free widths and orientation.
TRAP compares the orientation with the position angle modulo 360 deg and requires the
fit's y axis to be the long one, although the same ellipse can come out with its axes
swapped (orientation + 90 deg) or turned by 180 deg. `analyse_experiments.py` therefore
also tests a form of the rule that does not depend on the parametrisation. For the
tutorial candidate it does not change the verdict: its long axis is 57 deg from the
radial direction.

## What the runs test

`run_investigation.sh` runs these steps one after the other and logs each to
`$SPHERICAL_TUTORIAL_DIR/investigation_logs/`.

1. Preprocessing times on the idle machine for IRDIS at `NCPU`, `NCPU/2`, `NCPU/4` and
   4 cores (the first run's timings were taken under a load of 90 to 140).
2. Preprocessing of the 2017-09-27 sequence (OBS_ID 1837295, continuous waffle,
   89.6 min, 52.6 deg rotation), in which Maire et al. (2019) measured b at
   449.0 mas, PA 155.67 deg.
3. Component fraction 0.10 to 0.30, each with and without the known position, on the
   31 to 43 px annulus. Separates the fraction from the known-companion mask. The cell
   f0.20 with the known position repeats the regression run with the current code and
   TRAP 2.1.0.
4. Fake planets with b's K1/K2 contrast (6.95e-6, 1.09e-6) at 1 and 2 times that, at
   25, 37 and 55 px, b masked as known. Seven fakes per separation and brightness. This
   gives the rate at which the shape check rejects a real planet, and whether a relaxed
   rule would accept residuals.
5. The 2017 sequence: fraction 0.15 and 0.20 with and without the known position, and
   fakes at 37 px.
6. The search region: inner bound 1 instead of 31, outer bound 100, and the original
   1 to 200 px with the regression settings.
7. The analysis over all experiments.
8. Preprocessing times for IFS at `NCPU` and `NCPU/2` (set `RUN_IFS_TIMING=0` to skip).

Each experiment has its own folder `experiments/<source>/<name>/` with hard links to
the source run's observation tree, so no tutorial product is replaced. An injected
science cube is a new file and is deleted after its TRAP run.

## Running it

On the server, activate the `spherical` mamba environment, pull this branch and install it:

```bash
mamba activate spherical
git fetch && git checkout investigate/irdis-shape-check
pip install -e ".[pipeline]"
screen -S investigation
mamba activate spherical   # again: the shell inside screen may not inherit it
export SPHERICAL_TUTORIAL_DIR=/path/with/space/sphere_tutorials
export SPECIES_DIR=/path/to/species
export NCPU=48
bash docs/tutorials/runs/investigation/run_investigation.sh
```

`tail -f $SPHERICAL_TUTORIAL_DIR/investigation_logs/steps.log` shows the progress. The
whole run takes about eight hours and up to 70 GB, most of it in the preprocessing
timing runs, which can be deleted once measured.

## Copying the results back

The experiment folders hold hard links to the science cubes, so copy them without the
observation trees.

```bash
SRV=<host>:/path/with/space/sphere_tutorials
DST=~/data/sphere_tutorials
rsync -av --prune-empty-dirs --exclude='reduction/*/observation/' "$SRV/experiments/" "$DST/experiments/"
rsync -av "$SRV/investigation_logs/" "$SRV/investigation_csv" "$DST/"
rsync -av --prune-empty-dirs --exclude='*.fits' --exclude='*.npy' \
    --include='*/' --include='run_provenance.json' --include='*.jsonlog' --include='*.log' \
    --include='frames_info_*.csv' --exclude='*' "$SRV/51eri_irdis_2017/" "$DST/51eri_irdis_2017/"
```

Then rerun the analysis locally with `--stamps` to look at the maps.

## What follows from the results

- The tutorial settings: fraction, known position, search region. The tutorial is
  meant to show how to handle a known planet, so it passes b's position either way.
- Whether TRAP's shape limits or their parametrisation need to change (a TRAP issue
  with the injection statistics attached).
- Stage 2: rerun the IRDIS and IFS tutorials with the chosen settings on the idle
  machine, with an IFS quick-check annulus narrower than the 50 to 72 px regression
  one.

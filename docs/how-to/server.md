# Run on a server or cluster

After this guide you can run reductions unattended on a shared machine and control how
many CPUs they use.

## Set the number of CPUs

Give the reduction and TRAP the same budget, before you change any other TRAP setting.

```python
from trap.parameters import trap_config_for_ifs

from spherical.pipeline.pipeline_config import IFSReductionConfig

config = IFSReductionConfig()
config.set_ncpu(16)

trap_config = trap_config_for_ifs()
config.apply_trap_resources(trap_config)
```

`set_ncpu` sets every stage of the reduction. To give stages different budgets, set
the fields of `config.resources` instead, for example `config.resources.ncpu_trap = 32`,
and call `apply_trap_resources` after that. TRAP gets its CPU count only through
`apply_trap_resources`. [The configuration model](../concepts/configuration.md#cpus)
explains which setting wins.

## Choose where TRAP keeps its working data

TRAP writes the data its worker processes share into a scratch directory. If you set
none, it uses `/dev/shm` when that exists and has room, and the system's temporary
directory otherwise (`trap.parameters.resolve_scratch_dir`). `/dev/shm` is memory, which
counts against a job's memory limit on many clusters. Point TRAP at a disk instead,
after `apply_trap_resources`, which would reset it.

```python
trap_config.reduction = trap_config.reduction.merge(scratch_dir="/scratch/your_name/trap")
```

```{common-mistake}
A TRAP run that is killed, for example by a time limit, leaves its scratch folder
`trap_store_*` behind. In `/dev/shm` it keeps using memory until you delete it with
`rm -rf /dev/shm/trap_store_*`. Check for leftovers after a killed job.
```

## Make the script safe to run in the background

- Put the work in `main()` and call it under `if __name__ == "__main__":`, as in
  [Reduce a list of targets](survey.md). The IRDIS pre-processing starts new Python
  processes that import your script.
- Plots need no display. The reduction selects matplotlib's file-only backend itself.
- Turn off TRAP's progress bars, which only clutter a log file:

  ```python
  trap_config.processing = trap_config.processing.merge(use_progress_bar=False, verbose=False)
  ```

- Store the ESO password before the run if you need proprietary data
  ([Proprietary data](proprietary-data.md)).

## Start the run

On a machine you log in to, start the script so that it survives the end of your
session, and send its output to a file.

```bash
nohup python reduce_survey.py > reduce_survey.out 2>&1 &
```

`tmux` or `screen` work as well. On a cluster with SLURM, a batch script like this
one is a starting point. Adapt the resources to your data and your site's rules.

```bash
#!/bin/bash
#SBATCH --job-name=spherical
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=spherical_%j.out

python reduce_survey.py
```

Keep `set_ncpu` equal to `--cpus-per-task`. Activate the environment spherical is
installed in before `python`, or use `pixi run -e pipeline python reduce_survey.py`
from a pixi checkout.

## Follow the run

Each observation writes its own log, and the summary tools read them all.
[Monitor runs](monitor.md) shows how to see which targets finished and which failed.
If a job stops early, start the same script again. Finished steps are skipped
([Re-run part of a reduction](rerun.md)).

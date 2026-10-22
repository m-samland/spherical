"""Reduce 51 Eri, 2017-09-27 (IRDIS DB_K12, OBS_ID 1837295), as a check of the tutorial.

The 2017 sequence (198.C-0209(J), continuous waffle, 89.6 min, 52.6 deg rotation) is
deeper than the 2015 one the tutorial uses, and Maire et al. (2019, Table A.1) measured
51 Eri b in it at 449.0 +- 2.9 mas, PA 155.67 +- 0.37 deg (K1). It uses the settings of
examples/irdis_reduction_template.py with the tutorial directories:

    python docs/tutorials/runs/51eri_irdis_2017.py --ncpu 8 --no-trap

See docs/tutorials/runs/investigation/README.md.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "examples")]
import irdis_reduction_template as template  # noqa: E402
import tutorial_run as runner  # noqa: E402

LABEL = "51eri_irdis_2017"
OBS_ID = 1837295


def build(ncpu, label=LABEL):
    return runner.build(template, ncpu, label)


def source_hash():
    return runner.source_hash(template, build)


def main(argv=None):
    return runner.main(template, build, source_hash, LABEL, OBS_ID, argv)


if __name__ == "__main__":
    raise SystemExit(main())

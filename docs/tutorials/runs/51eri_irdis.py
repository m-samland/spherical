"""Reduce 51 Eri, 2015-09-24 (IRDIS DB_K12, OBS_ID 200363269), for the tutorial.

Uses the settings of examples/irdis_reduction_template.py with the tutorial
directories under $SPHERICAL_TUTORIAL_DIR (default ~/data/sphere_tutorials):

    python docs/tutorials/runs/51eri_irdis.py --ncpu 8

See docs/tutorials/runs/README.md for the whole procedure.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "examples")]
import irdis_reduction_template as template  # noqa: E402
import tutorial_run as runner  # noqa: E402

LABEL = "51eri_irdis"
OBS_ID = 200363269


def build(ncpu, label=LABEL):
    return runner.build(template, ncpu, label)


def source_hash():
    return runner.source_hash(template, build)


def main(argv=None):
    return runner.main(template, build, source_hash, LABEL, OBS_ID, argv)


if __name__ == "__main__":
    raise SystemExit(main())

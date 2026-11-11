"""Reduce 51 Eri, 2015-09-24 (IRDIS DB_K12, OBS_ID 200363269), for the tutorial.

Uses the settings of examples/irdis_reduction_template.py with the tutorial
directories under $SPHERICAL_TUTORIAL_DIR (default ~/data/sphere_tutorials), and the
TRAP settings of the 51 Eri regression test for the known planet:

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
    config, trap_config = runner.build(template, ncpu, label)
    # 51 Eri b is known, so TRAP searches the annulus around it and keeps the planet out
    # of the noise model for every other position (the settings of the 51 Eri
    # regression test in tests/regression/run_51eri_irdis_reference.py).
    trap_config.reduction = trap_config.reduction.merge(
        search_region_inner_bound=31,
        search_region_outer_bound=43,
        yx_known_companion_position=[-35.95, -8.43],  # (y, x) offset from the star in px
    )
    trap_config.processing = trap_config.processing.merge(temporal_components_fraction=[0.2])
    # With the background noise in the IRDIS ivar, 51 Eri b peaks at a norm-SNR of about
    # 4.6, below trap's default thresholds, so the regression run lowers them.
    trap_config.detection = trap_config.detection.merge(candidate_threshold=4.0, detection_threshold=4.5)
    return config, trap_config


def source_hash():
    return runner.source_hash(template, build)


def main(argv=None):
    return runner.main(template, build, source_hash, LABEL, OBS_ID, argv)


if __name__ == "__main__":
    raise SystemExit(main())

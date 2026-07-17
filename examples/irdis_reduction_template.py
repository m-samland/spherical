"""IRDIS reduction — Phase 1 driver (download-only).

Downloads the raw science + calibration files (``CORO``, ``CENTER``, ``FLUX``,
``FLAT``, ``BG_SCIENCE``) for the selected IRDIS observations from the ESO
archive. Nothing else runs yet: calibration + preprocessing are wired in later
phases. Use this template to pull the reference dataset that subsequent phases
will validate against.
"""
from pathlib import Path

from astropy.table import Table

from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IRDISReductionConfig

target_list = ["* bet Pic"]

# =================== CONFIGURATION ===================
config = IRDISReductionConfig()

config.set_ncpu(4)

# Phase 1: only download_data is wired. Disable every other step by starting
# from a fully-off state and enabling just the download.
config.steps.disable_all_ifs_steps()
config.steps = config.steps.merge(
    download_data=True,
    run_trap_reduction=False,
    run_trap_detection=False,
)

# ===== ESO archive credentials =====
config.preprocessing = config.preprocessing.merge(
    eso_username=None,
    store_password=False,
    delete_password_after_reduction=False,
)

# ===== Directory layout =====
config.directories.base_path = Path.home() / "data/sphere"
config.directories.raw_directory = config.directories.base_path / "data"
config.directories.reduction_directory = config.directories.base_path / "reduction"

database_directory = Path.home() / "data/sphere/database"

instrument = "irdis"

table_of_observations = Table.read(
    database_directory / f"table_of_observations_{instrument}.fits"
)
table_of_files = Table.read(
    database_directory / f"table_of_files_{instrument}.csv"
)

# ---------------------Database setup-----------------------------------------#
database = SphereDatabase(
    table_of_observations, table_of_files, instrument=instrument
)

observation_table = database.filter(
    target_list=target_list,
    TOTAL_EXPTIME_SCI=(">", 30),
    DEROTATOR_MODE="PUPIL",
    HCI_READY=True,
)
print(observation_table)

observations = database.retrieve_observation_metadata(observation_table)


def main():
    execute_targets(observations=observations, config=config)


if __name__ == "__main__":
    main()

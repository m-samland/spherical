"""Produce the 51 Eri IRDIS reference reduction behind the frozen astrometry baseline.

The exact configuration of ``data/51eri_astrometry_benchmark.md`` §1, so the baseline can
be reproduced without editing ``examples/irdis_reduction_template.py`` by hand. Runs the
full chain (download → calibration → preprocessing → centering/photometry → TRAP
reduction + detection) for the single DB_K12 observation of 2015-09-24, then writes
``provenance.json`` into the TRAP result folder.

Every step is forced, so existing products under ``REDUCTION_DIRECTORY`` are recomputed
in place; only raw files already on disk are not downloaded again.

    python tests/regression/run_51eri_irdis_reference.py

Run it in an environment with non-editable installs (see the benchmark doc §9): the
provenance record reads versions and commits from the installed distributions.
"""
import datetime
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from astropy.table import Table
from trap.parameters import trap_config_for_irdis

from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IRDISReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations

BASE_PATH = Path.home() / "data/sphere"
REDUCTION_DIRECTORY = BASE_PATH / "reduction"
SPECIES_DATABASE_DIRECTORY = BASE_PATH / "species"
NCPU = 4

# Benchmark doc §1.
TARGET_LIST = ["51 Eridani"]
NIGHT_START = "2015-09-24"
SEARCH_REGION_INNER_BOUND = 31
SEARCH_REGION_OUTER_BOUND = 43
YX_KNOWN_COMPANION_POSITION = [-35.95, -8.43]
TEMPORAL_COMPONENTS_FRACTION = [0.2]  # trap's default is [0.15]

PROVENANCE_PACKAGES = [
    "spherical", "trap", "charis", "species",
    "numpy", "scipy", "astropy", "photutils", "scikit-image", "numba", "pandas",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_record(name: str) -> dict:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None}
    record = {"version": dist.version}
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        info = json.loads(direct_url)
        record["url"] = info.get("url")
        record["vcs_info"] = info.get("vcs_info")
        # An editable install's version string is stamped at install time and goes stale.
        record["editable"] = info.get("dir_info", {}).get("editable", False)
    return record


def main():
    started = datetime.datetime.now(datetime.timezone.utc)

    config = IRDISReductionConfig()
    config.set_ncpu(NCPU)
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()
    config.steps = config.steps.merge(
        download_data=True,
        irdis_calibration=True,
        preprocess_irdis=True,
        compute_frames_info=True,
        cube_header_update=True,
        find_centers=True,
        process_extracted_centers=True,
        plot_image_center_evolution=True,
        calibrate_spot_photometry=True,
        calibrate_flux_psf=True,
        spot_to_flux=True,
        run_trap_reduction=True,
        run_trap_detection=True,
        # A bool, not a set of step names: run_trap validates a set against the IFS
        # step list, which rejects IRDIS-only names such as "preprocess_irdis".
        force=True,
    )
    config.preprocessing = config.preprocessing.merge(
        eso_username=None,
        store_password=False,
        delete_password_after_reduction=False,
    )
    config.pass_inverse_variance_to_trap = True
    config.pass_center_outliers_as_bad_frames_to_trap = True
    config.pass_amplitude_modulation_to_trap = True

    config.directories.base_path = BASE_PATH
    config.directories.raw_directory = BASE_PATH / "data"
    config.directories.reduction_directory = REDUCTION_DIRECTORY

    database_directory = resolve_database_dir(default=BASE_PATH / "database")
    observations_file = database_directory / "table_of_observations_irdis.fits"
    files_file = database_directory / "table_of_files_irdis.csv"
    database = SphereDatabase(
        Table.read(observations_file), Table.read(files_file), instrument="irdis"
    )
    observation_table = database.filter(
        target_list=TARGET_LIST,
        TOTAL_EXPTIME_SCI=(">", 30),
        DEROTATOR_MODE="PUPIL",
        HCI_READY=True,
        NIGHT_START=NIGHT_START,
    )
    if len(observation_table) != 1:
        raise RuntimeError(f"expected one observation, got {len(observation_table)}")
    print(observation_table)
    observations = database.retrieve_observation_metadata(observation_table)

    execute_targets(observations=observations, config=config)

    trap_config = trap_config_for_irdis()
    config.apply_trap_resources(trap_config)
    trap_config.reduction = trap_config.reduction.merge(
        search_region_inner_bound=SEARCH_REGION_INNER_BOUND,
        search_region_outer_bound=SEARCH_REGION_OUTER_BOUND,
        yx_known_companion_position=YX_KNOWN_COMPANION_POSITION,
    )
    trap_config.processing = trap_config.processing.merge(
        temporal_components_fraction=TEMPORAL_COMPONENTS_FRACTION,
    )

    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=SPECIES_DATABASE_DIRECTORY,
    )

    result_folders = [
        p
        for p in (REDUCTION_DIRECTORY / "IRDIS" / "trap").glob(f"*/DB_K12/{NIGHT_START}")
        if "51_Eri" in str(p)
    ]
    if len(result_folders) != 1:
        raise RuntimeError(f"expected one TRAP result folder, got {result_folders}")
    # run_trap_on_observations logs a crash report instead of raising, so check that
    # this run actually produced the table before stamping provenance on the folder.
    table = result_folders[0] / "template_matching" / "overall_validated_companion_detections.csv"
    if not table.exists() or table.stat().st_mtime < started.timestamp():
        raise RuntimeError(f"{table} was not written by this run; see the crash report")

    provenance = {
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "driver": {
            "path": "tests/regression/run_51eri_irdis_reference.py",
            "sha256": _sha256(Path(__file__)),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "ncpu": NCPU,
        "packages": {name: _package_record(name) for name in PROVENANCE_PACKAGES},
        "database_tables": {
            observations_file.name: _sha256(observations_file),
            files_file.name: _sha256(files_file),
        },
        "settings": {
            "night_start": NIGHT_START,
            "search_region_inner_bound": SEARCH_REGION_INNER_BOUND,
            "search_region_outer_bound": SEARCH_REGION_OUTER_BOUND,
            "yx_known_companion_position": YX_KNOWN_COMPANION_POSITION,
            "temporal_components_fraction": TEMPORAL_COMPONENTS_FRACTION,
        },
    }
    (result_folders[0] / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


# Phase 4 spawns worker processes that re-import this file; keep everything in main().
if __name__ == "__main__":
    main()

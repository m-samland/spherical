"""Produce the 51 Eri IFS reference reduction behind the frozen astrometry baseline.

The IFS counterpart of ``run_51eri_irdis_reference.py``, with the configuration of
``data/51eri_astrometry_benchmark.md`` §8b-bis. Runs the full chain (download →
wavelength calibration → CHARIS cube extraction → bundling → centering/photometry → TRAP
reduction + detection) for the single OBS_H observation of 2015-09-24, then writes
``provenance.json`` into the TRAP result folder.

Every step is forced, so existing products under ``REDUCTION_DIRECTORY`` are recomputed
in place; only raw files already on disk are not downloaded again.

    python tests/regression/run_51eri_ifs_reference.py

Two environment variables adapt the run to another machine without editing this file, so
the driver sha256 in the provenance record stays comparable:

- ``SPHERICAL_BASE_PATH``: directory holding ``data/``, ``reduction/``, ``database/`` and
  ``species/`` (default ``~/data/sphere``). ``$SPHERICAL_DATABASE_DIR`` still overrides the
  database location.
- ``SPHERICAL_NCPU``: worker count for extraction, centering and TRAP (default 4).

Run it in an environment with non-editable installs (see the benchmark doc §9): the
provenance record reads versions and commits from the installed distributions.
"""
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from astropy.table import Table
from trap.parameters import trap_config_for_ifs

from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase
from spherical.pipeline.ifs_reduction import execute_targets
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.run_trap import run_trap_on_observations

BASE_PATH = Path(os.environ.get("SPHERICAL_BASE_PATH", Path.home() / "data/sphere")).expanduser()
REDUCTION_DIRECTORY = BASE_PATH / "reduction"
SPECIES_DATABASE_DIRECTORY = BASE_PATH / "species"
NCPU = int(os.environ.get("SPHERICAL_NCPU", 4))

# Benchmark doc §8b-bis. The annulus brackets the planet at 61 px; IFS pixels are
# 7.46 mas, so it is wider in pixels than the IRDIS one.
TARGET_LIST = ["51 Eridani"]
NIGHT_START = "2015-09-24"
OBS_MODE = "OBS_H"
SEARCH_REGION_INNER_BOUND = 50
SEARCH_REGION_OUTER_BOUND = 72
YX_KNOWN_COMPANION_POSITION = [-59.03, -13.84]
TEMPORAL_COMPONENTS_FRACTION = [0.15]
SEARCH_RADIUS = 15
CANDIDATE_THRESHOLD = 4.75
DETECTION_THRESHOLD = 5.0

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

    config = IFSReductionConfig()
    config.set_ncpu(NCPU)
    config.steps.disable_all_ifs_steps()
    config.steps = config.steps.merge(
        download_data=True,
        reduce_calibration=True,
        extract_cubes=True,
        bundle_output=True,
        bundle_hexagons=False,
        bundle_residuals=False,
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
        force=True,
    )
    config.preprocessing = config.preprocessing.merge(
        eso_username=None,
        store_password=False,
        delete_password_after_reduction=False,
    )
    # 51 Eri 2015-09-24 is a coronagraphic sequence with CENTER frames, not continuous
    # waffle, so the two waffle-only inputs would be ignored anyway.
    config.pass_inverse_variance_to_trap = True
    config.pass_center_outliers_as_bad_frames_to_trap = False
    config.pass_amplitude_modulation_to_trap = False

    config.directories.base_path = BASE_PATH
    config.directories.raw_directory = BASE_PATH / "data"
    config.directories.reduction_directory = REDUCTION_DIRECTORY

    database_directory = resolve_database_dir(default=BASE_PATH / "database")
    observations_file = database_directory / "table_of_observations_ifs.fits"
    files_file = database_directory / "table_of_files_ifs.csv"
    database = SphereDatabase(
        Table.read(observations_file), Table.read(files_file), instrument="ifs"
    )
    observation_table = database.filter(
        target_list=TARGET_LIST,
        TOTAL_EXPTIME_SCI=(">", 30),
        DEROTATOR_MODE="PUPIL",
        HCI_READY=True,
        NIGHT_START=NIGHT_START,
        FILTER=OBS_MODE,
    )
    if len(observation_table) != 1:
        raise RuntimeError(f"expected one observation, got {len(observation_table)}")
    print(observation_table)
    observations = database.retrieve_observation_metadata(observation_table)

    execute_targets(observations=observations, config=config)

    trap_config = trap_config_for_ifs()
    config.apply_trap_resources(trap_config)
    trap_config.reduction = trap_config.reduction.merge(
        search_region_inner_bound=SEARCH_REGION_INNER_BOUND,
        search_region_outer_bound=SEARCH_REGION_OUTER_BOUND,
        yx_known_companion_position=YX_KNOWN_COMPANION_POSITION,
    )
    trap_config.detection = trap_config.detection.merge(
        search_radius=SEARCH_RADIUS,
        candidate_threshold=CANDIDATE_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
        use_spectral_correlation=False,
    )
    trap_config.processing = trap_config.processing.merge(
        temporal_components_fraction=TEMPORAL_COMPONENTS_FRACTION,
        verbose=False,
        use_progress_bar=False,
    )

    run_trap_on_observations(
        observations=observations,
        trap_config=trap_config,
        reduction_config=config,
        species_database_directory=SPECIES_DATABASE_DIRECTORY,
    )

    result_folders = [
        p
        for p in (REDUCTION_DIRECTORY / "IFS" / "trap").glob(f"*/{OBS_MODE}/{NIGHT_START}")
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
            "path": "tests/regression/run_51eri_ifs_reference.py",
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
            "obs_mode": OBS_MODE,
            "search_region_inner_bound": SEARCH_REGION_INNER_BOUND,
            "search_region_outer_bound": SEARCH_REGION_OUTER_BOUND,
            "yx_known_companion_position": YX_KNOWN_COMPANION_POSITION,
            "temporal_components_fraction": TEMPORAL_COMPONENTS_FRACTION,
            "search_radius": SEARCH_RADIUS,
            "candidate_threshold": CANDIDATE_THRESHOLD,
            "detection_threshold": DETECTION_THRESHOLD,
        },
    }
    (result_folders[0] / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


# Cube extraction and the center fit spawn worker processes that re-import this file;
# keep everything in main().
if __name__ == "__main__":
    main()

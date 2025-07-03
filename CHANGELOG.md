# Changelog

All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

### ✨ Added
- **Typed dataclass configuration system** – Refactored the IFS reduction pipeline configuration from plain dictionaries to a comprehensive typed dataclass-based architecture. The new system provides type safety, IDE autocompletion, intelligent defaults, and better maintainability while maintaining backward compatibility through `as_plain_dicts()` method ([@m-samland](https://github.com/m-samland)).
- **Centralized logging infrastructure** – Implemented uniform logging schema across all IFS pipeline steps with `@optional_logger` decorator, structured logging with automatic context injection, and multiprocessing-safe `QueueHandler` mechanism. All pipeline steps now support consistent, testable, and aggregatable logging ([@m-samland](https://github.com/m-samland)).
- **Pipeline monitoring scripts** – Added installable command-line tools `crash_reports` and `reduction_status` to aggregate and summarize pipeline execution across large reduction campaigns. Scripts parse structured logs and crash reports to provide dataset completion status, exception frequency analysis, and CSV export capabilities ([@m-samland](https://github.com/m-samland)).
- **Multi-observation wrapper function** – Added `execute_targets()` wrapper function to `ifs_reduction.py` that processes multiple SPHERE observations sequentially using the same configuration, simplifying batch processing workflows and removing explicit loops from user scripts ([@m-samland](https://github.com/m-samland)).
- **TRAP post-processing wrapper functions** – Added `run_trap_on_observation()` and `run_trap_on_observations()` wrapper functions to `pipeline/run_trap.py` module, providing a consistent modular interface for TRAP post-processing that mirrors the IFS reduction pipeline structure. Refactored inline TRAP processing code into reusable functions, improving maintainability and code organization ([@m-samland](https://github.com/m-samland)).
- **Comprehensive documentation** – Added NumPy-style docstrings throughout the `ifs_reduction.py` module following astronomical software documentation standards, including scientific context, parameter units, wavelength specifications, and coordinate system references ([@m-samland](https://github.com/m-samland)).

### Changed
- **Pipeline configuration architecture** – Introduced `CalibrationConfig`, `ExtractionConfig`, `PreprocConfig`, `DirectoryConfig`, and `Resources` dataclasses with merge functionality, centralized CPU allocation, and automatic path resolution. Factory method `defaultIFSReduction()` provides easy default configuration creation ([@m-samland](https://github.com/m-samland)).
- **Pipeline logging standardization** – Refactored all pipeline steps to use centralized logging with automatic injection of static context fields (`target`, `band`, `night`), structured status logging (`status`: `"success"`/`"failed"`), and eliminated `print()` statements in favor of proper log levels. Enhanced multiprocessing safety and debugging capabilities ([@m-samland](https://github.com/m-samland)).
- **Database class naming** – Updated `Sphere_database` to `SphereDatabase` following PEP8 naming conventions for improved code consistency ([@m-samland](https://github.com/m-samland)).
- **Modular pipeline architecture** – Completely restructured the IFS data reduction pipeline into discrete, self-contained modules located in `pipeline/steps/`. Each processing step (wavelength calibration, cube extraction, astrometric calibration, etc.) is now an independent module with a single function call containing all required logic. This modularization significantly improves code maintainability, enables comprehensive unit testing, and provides a future-proof architecture for pipeline extensions ([@m-samland](https://github.com/m-samland)).

## [2.0.0] – IRDIS Support and Pipeline Enhancements (2025-05-18)

### ✨ Added
- **IRDIS dual-band imaging data support** – The observation database now includes VLT/SPHERE IRDIS dual-band imaging (DBI) sequences for seamless querying and retrieval via the pipeline ([@m-samland](https://github.com/m-samland)) ([#53](https://github.com/m-samland/spherical/issues/53)).
- **IRDIS polarimetry data support** – Added SPHERE/IRDIS dual-polarization imaging (DPI) observations to the database, enabling search and download of IRDIS polarimetric data through the same interface ([@m-samland](https://github.com/m-samland)) ([#49](https://github.com/m-samland/spherical/issues/49)).
- **Unified data download module** – The data download process was refactored into an independent module that handles both **IFS** and **IRDIS** instrument modes consistently. It now supports proper folder management, avoids redundant downloads, and permits custom keyword filtering for additional file types ([@m-samland](https://github.com/m-samland)) ([#60](https://github.com/m-samland/spherical/issues/60)).

### Changed
- **Refactored data download architecture** – The download functionality was overhauled to robustly handle multiple instrument modes (IFS and IRDIS) and improve reliability for large batch fetches ([@m-samland](https://github.com/m-samland)) ([#60](https://github.com/m-samland/spherical/issues/60)).
- **New sequence readiness flag** – Replaced the `FAILED_SEQ` status flag with a clearer `HCI_READY` flag to mark sequences that are ready for high-contrast imaging analysis in the database ([@m-samland](https://github.com/m-samland)) ([#56](https://github.com/m-samland/spherical/issues/56)).
- **Optimized SIMBAD cross-matching** – Pre-selects candidate targets based on sky position (Healpix) before querying SIMBAD, reducing false matches and lowering query overhead ([@m-samland](https://github.com/m-samland)) ([#54](https://github.com/m-samland/spherical/issues/54)).
- **Incremental header saving** – During database construction, header tables are now saved incrementally to avoid memory issues and prevent data loss if a process is interrupted ([@m-samland](https://github.com/m-samland)) ([#52](https://github.com/m-samland/spherical/issues/52)).
- **Updated external pipeline dependency** – Pointed the SPHERE/CHARIS IFS spectral extraction pipeline to its latest **main** branch for long-term stability improvement ([@m-samland](https://github.com/m-samland)) ([#47](https://github.com/m-samland/spherical/issues/47)).
- **Module renaming for clarity** – Renamed the `sphere_database` module to `database` to simplify the package structure and usage ([@m-samland](https://github.com/m-samland)) ([#44](https://github.com/m-samland/spherical/issues/44)).
- **Metadata computation refactor** – Moved calculation of observational metadata (e.g. exposure times, parallactic rotation) into the database module for a cleaner pipeline workflow ([@m-samland](https://github.com/m-samland)) ([#42](https://github.com/m-samland/spherical/issues/42)).

### Fixed
- **Batch table overwrite bug** – Fixed an issue where creating the file table in batches could overwrite results from earlier batches ([@m-samland](https://github.com/m-samland)) ([#50](https://github.com/m-samland/spherical/issues/50)).
- **Parallactic angle calculation (important)** – Corrected the `ROTATION` (parallactic angle change) values in observation summary tables to reflect true field rotation ([@m-samland](https://github.com/m-samland)) ([#46](https://github.com/m-samland/spherical/issues/46)).
- **PSF extraction in unocculted frames** – Resolved a bug causing the PSF-finding routine to fail on non-coronagraphic images when the telescope pointing offset changed between frames ([@m-samland](https://github.com/m-samland)) ([#45](https://github.com/m-samland/spherical/issues/45)).
- **Spectral cube assembly** – The cube building step now handles incorrect NDIT header values gracefully, instead of aborting the process ([@m-samland](https://github.com/m-samland)) ([#43](https://github.com/m-samland/spherical/issues/43)).
- **Run extract_cube without previous steps** – Fixed a file path renaming issue that would prevent the `extract_cube`-step to run, when not running the previous steps in the same session ([@m-samland](https://github.com/m-samland)) ([#41](https://github.com/m-samland/spherical/issues/41)).
- **Timing table overwrite** – Ensured that the `compute_times` function no longer overwrites its input timing table, preserving original data ([@m-samland](https://github.com/m-samland)) ([#40](https://github.com/m-samland/spherical/issues/40)).


## [1.1.1] - 2025-04-10

### ✨ Added
- Documented that Python 3.13 is not yet supported when installing the pipeline dependencies (in the README). ([@m-samland](https://github.com/m-samland)) ([#32](https://github.com/m-samland/spherical/pull/32))
- Added multi-processing to find the star center using satellite spots for each frame. ([@m-samland](https://github.com/m-samland)) ([#31](https://github.com/m-samland/spherical/pull/31))

### Changed
- Improved the plot of star center evolution over time and wavelength; this is now an independent step. ([@m-samland](https://github.com/m-samland)) ([#38](https://github.com/m-samland/spherical/pull/38))
- Moved `find_star` to its own module. ([@m-samland](https://github.com/m-samland)) ([#37](https://github.com/m-samland/spherical/pull/37))
- Refactored `compute_times` in the pipeline to include `DIT_DELAY`. ([@m-samland](https://github.com/m-samland)) ([#36](https://github.com/m-samland/spherical/pull/36))
- Renamed `calibrate_center` to `find_star`. ([@m-samland](https://github.com/m-samland)) ([#22](https://github.com/m-samland/spherical/pull/22))

### Fixed
- Fixed multiple bugs to enable running each step of the IFS reduction template. ([@m-samland](https://github.com/m-samland)) ([#27](https://github.com/m-samland/spherical/pull/27), [#29](https://github.com/m-samland/spherical/pull/29))

## [1.1.0] - 2025-04-04

### ✨ Added
- Added documentation to the high-level database functions describing the content of all tables. ([@m-samland](https://github.com/m-samland)) ([#22](https://github.com/m-samland/spherical/pull/22))
- Added file size estimates for each SPHERE FITS file to support download size estimates per observation sequence. ([@m-samland](https://github.com/m-samland)) ([#16](https://github.com/m-samland/spherical/pull/16))
- Added progress bar for database table creation. ([@lwelzel](https://github.com/lwelzel)) ([#2](https://github.com/m-samland/spherical/pull/2))
- Added support for batched queries. ([@lwelzel](https://github.com/lwelzel)) ([#2](https://github.com/m-samland/spherical/pull/2))
- Added changelog. ([@m-samland](https://github.com/m-samland)) ([#6](https://github.com/m-samland/spherical/pull/6))
- Added CI and `ruff` linting workflow. ([@m-samland](https://github.com/m-samland)) ([#5](https://github.com/m-samland/spherical/pull/5))
- Added basic end-to-end tests for the database. ([@m-samland](https://github.com/m-samland)) ([#4](https://github.com/m-samland/spherical/pull/4))
- Added issue and PR templates. ([@m-samland](https://github.com/m-samland)) ([#3](https://github.com/m-samland/spherical/pull/3))

### Changed
- Renamed "master file table" to simply "file table". ([@m-samland](https://github.com/m-samland)) ([#22](https://github.com/m-samland/spherical/pull/22))
- Filtered SIMBAD stars associated with observation sequences to require proper motion. Excluded stars with parallaxes implying distances greater than 1 kpc (unlikely to be direct imaging targets). ([@m-samland](https://github.com/m-samland)) ([#13](https://github.com/m-samland/spherical/pull/13))
- Switched to using an `astropy` function for proper motion propagation. ([@m-samland](https://github.com/m-samland)) ([#12](https://github.com/m-samland/spherical/pull/12))
- Separated pipeline dependencies from the database. To analyze data, use:  
  ```bash
  pip install ".[pipeline]"
  ```  
  ([@m-samland](https://github.com/m-samland)) ([#9](https://github.com/m-samland/spherical/pull/9))
- Added summary information (e.g., total exposure time, parallactic angle change) to the observation sequence table with values rounded to sensible precision. ([@m-samland](https://github.com/m-samland)) ([#15](https://github.com/m-samland/spherical/pull/15))

### Fixed
- Fixed compatibility with newer `astroquery` versions. ([@lwelzel](https://github.com/lwelzel), [@m-samland](https://github.com/m-samland)) ([#2](https://github.com/m-samland/spherical/pull/2), [#14](https://github.com/m-samland/spherical/pull/14))
- Fixed an issue where the number of integrations (`NDIT`) was read under two different names, causing downstream errors. ([@m-samland](https://github.com/m-samland)) ([#10](https://github.com/m-samland/spherical/pull/10))

---

## [1.0.0] - 2024-12-27

### Added
- Initial release.

### Changed
- Initial implementation of core functionality.

### Fixed
- No known issues.

[Unreleased]: https://github.com/m-samland/spherical/compare/v1.1.1...HEAD  
[1.1.1]: https://github.com/m-samland/spherical/compare/v1.1.0...v1.1.1  
[1.1.0]: https://github.com/m-samland/spherical/compare/v1.0.0...v1.1.0  
[1.0.0]: https://github.com/m-samland/spherical/releases/tag/v1.0.0

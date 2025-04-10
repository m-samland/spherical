# Changelog

All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

## [1.1.1] - 2025-04-10

### Added
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

### Added
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

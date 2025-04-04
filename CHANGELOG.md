# Changelog

All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/) format.

---

## [Unreleased]

## [1.1.0] - 2025-04-04

### Added
- Added documentation to the high-level database functions describing the content of all tables. ([@m-samland](https://github.com/m-samland)) (#22)
- Added file size estimate for each SPHERE FITS file. This allows us to estimate the size of downloading a specific SPHERE observation sequence. ([@m-samland](https://github.com/m-samland)) (#16)
- Added progressbar for database table creation. ([#2](https://github.com/m-samland/soherical/pull/2), [@lwelzel](https://github.com/lwelzel))
- Added support for batched queries. ([#2](https://github.com/m-samland/soherical/pull/2), [@lwelzel](https://github.com/lwelzel))
- Added changelog. ([@m-samland](https://github.com/m-samland)) (#6)
- Added CI and ruff linting workflow. ([@m-samland](https://github.com/m-samland)) (#5)
- Added basic end-to-end testing for database. ([@m-samland](https://github.com/m-samland)) (#4)
- Added issue and PR templates. ([@m-samland](https://github.com/m-samland)) (#3)

### Changed
- Renamed master file table to file table. ([@m-samland](https://github.com/m-samland)) (#22)
- Filter Simbad stars associated with each observation sequences to require proper-motion information and filter out stars with parallaxes corresponding to very large distances unlikely to be direct imaging targets (default: >1 kpc). ([@m-samland](https://github.com/m-samland)) (#13)
- Use astropy function for proper motion propagation. ([@m-samland](https://github.com/m-samland)) (#12)
- Separate pipeline dependencies from database. If you want to analyze data use: pip install ".[pipeline]" ([@m-samland](https://github.com/m-samland)) (#9)
- Useful information added to the table of observation sequences, such as overall parallactic angle change and total exposure time, are rounded to sensible digits. ([@m-samland](https://github.com/m-samland)) (#15)

### Fixed
- Bug fixes to support newer astroquery version. ([#2](https://github.com/m-samland/soherical/pull/2), [@lwelzel](https://github.com/lwelzel)) and ([@m-samland](https://github.com/m-samland)) (#14)
- Number of integration (NDIT) keyword was read in twice under different names resulting in problems down-stream. ([@m-samland](https://github.com/m-samland)) (#10)

---

## [1.0.0] - 2024-12-27

### Added
- Initial release.

### Changed
- Initial implementation of project functionality.

### Fixed
- No known bugs yet.

[Unreleased]: https://github.com/m-samland/spherical/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/m-samland/spherical/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/m-samland/spherical/releases/tag/v1.0.0
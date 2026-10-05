<p align="center">
  <img src="https://raw.githubusercontent.com/m-samland/spherical/main/assets/logos/spherical_banner.png" alt="spherical: VLT/SPHERE data analysis tools" width="65%">
</p>

# spherical: VLT/SPHERE observation database and IFS + IRDIS pipeline

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-brightgreen.svg)](https://github.com/m-samland/spherical)
[![License](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/m-samland/spherical/actions/workflows/ci.yml/badge.svg)](https://github.com/m-samland/spherical/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/spherical-hci/badge/?version=latest)](https://spherical-hci.readthedocs.io/en/latest/)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2509.08044-blue.svg)](https://doi.org/10.48550/arXiv.2509.08044)

**spherical** finds VLT/SPHERE observations in the ESO archive and reduces IFS and
IRDIS data to companion spectra and astrometry. Its database describes every SPHERE
observation sequence in the archive, matched to Gaia DR3 and MOCA. Its pipeline takes
a sequence from download through calibration and TRAP post-processing to the
photometry and astrometry of companions.

**Documentation: [spherical-hci.readthedocs.io](https://spherical-hci.readthedocs.io/en/latest/)**

## Installation

Python 3.11 or newer on Linux or macOS.

```bash
# database only
pip install git+https://github.com/m-samland/spherical.git

# full IFS and IRDIS pipeline
pip install "spherical[pipeline] @ git+https://github.com/m-samland/spherical.git"
```

The package named `spherical` on PyPI is unrelated. Pixi and development installs are
described in the [installation guide](https://spherical-hci.readthedocs.io/en/latest/getting-started/installation.html).

## Quick start

```bash
spherical-sync-tables --dest ~/data/sphere/database
export SPHERICAL_DATABASE_DIR=~/data/sphere/database
```

```python
from astropy.table import Table
from spherical.database.paths import resolve_database_dir
from spherical.database.sphere_database import SphereDatabase

database_dir = resolve_database_dir(default="~/data/sphere/database")
db = SphereDatabase(Table.read(database_dir / "table_of_observations_ifs.fits"),
                    Table.read(database_dir / "table_of_files_ifs.csv"), instrument="ifs")
print(db.filter(target_list=["51 Eri"], usable_only=True))
```

To reduce data, start from
[`examples/ifs_reduction_template.py`](https://github.com/m-samland/spherical/blob/develop/examples/ifs_reduction_template.py)
or
[`examples/irdis_reduction_template.py`](https://github.com/m-samland/spherical/blob/develop/examples/irdis_reduction_template.py).

## Documentation

- [Getting started](https://spherical-hci.readthedocs.io/en/latest/getting-started/index.html): installation, the database and a first query
- [Reference](https://spherical-hci.readthedocs.io/en/latest/reference/index.html): conventions, configuration, TRAP settings, pipeline steps, command-line tools and the Python API
- [Contributing](https://spherical-hci.readthedocs.io/en/latest/contributing/index.html): development setup, tests and the writing guide
- [Changelog](https://spherical-hci.readthedocs.io/en/latest/project/changelog.html)

## Citation

If spherical supports your research, please cite

- spherical: [Samland (2025)](https://arxiv.org/abs/2509.08044)
- the IFS cube extraction: [Samland et al. (2022)](https://doi.org/10.1051/0004-6361/202244587)
- TRAP post-processing: [Samland et al. (2021)](https://doi.org/10.1051/0004-6361/201937308)
- the species package, used for spectral templates: [Stolker et al. (2020)](https://doi.org/10.1051/0004-6361/201937159)

The calibration of both pipelines and parts of the IRDIS reduction build on A. Vigan's
SPHERE tools ([repository](https://github.com/avigan/SPHERE),
[Vigan 2020](https://ascl.net/2009.002)); please acknowledge them as well. See
[Citing](https://spherical-hci.readthedocs.io/en/latest/project/citing.html) for
BibTeX.

## License

[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)

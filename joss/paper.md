---
title: 'spherical: A Comprehensive Database and Automated Pipeline for VLT/SPHERE High-Contrast Imaging'
tags:
  - Python
  - astronomy
  - exoplanets
  - protoplanetary disks
  - circumstellar disks
  - high-contrast imaging
  - direct imaging
authors:
  - name: Matthias Samland
    orcid: 0000-0002-0101-8814
    affiliation: 1

affiliations:
  - name: Max-Planck-Institut für Astronomie (MPIA), Königstuhl 17, 69117 Heidelberg, Germany
    index: 1

date: 23 July 2025
bibliography: paper.bib
---

### Summary

The Spectro-Polarimetric High-contrast Exoplanet REsearch instrument (SPHERE; @Beuzit:2019) at the Very Large Telescope (VLT) is among the most advanced ground-based instruments for coronagraphic imaging of exoplanets and circumstellar disks in the optical and near-infrared. Over the last decade, SPHERE has contributed to over 400 [publications](https://telbib.eso.org/) and led major surveys of exoplanets and disks (SHINE; @Chauvin:2017; @Chomez:2025), revealing planetary companions and disk structures around nearby stars.

The spherical software package offers both a ready-to-use, searchable database of all SPHERE observations from the ESO archive—available via [Zenodo](https://doi.org/10.5281/zenodo.15147730)—and a framework for generating and updating this database using the latest archive contents, allowing users to explore, download, and analyze high-contrast imaging datasets efficiently.
The SPHERE instrument consists of three sub-instruments. The InfraRed Dual-band Imager and Spectrograph (IRDIS; @Dohlen:2008; @Vigan:2010), the Integral Field Spectrograph (IFS; @Claudi:2008; @Mesa:2015), and the optical Zurich Imaging Polarimeter (ZIMPOL; @Schmid:2018).
As of May 2025, the spherical database includes ~6000 IRDIS observations in the dual-band imaging (DBI) mode, ~1000 in the IRDIS dual-beam polarimetric imaging (DPI; @deBoer:2020; @vanHolstein:2020) mode, and ~4500 in the IFS mode. The ZIMPOl instrument, IRDIS long-slit spectroscopy (LSS; @Vigan:2008) mode, and Sparse Aperture Masking (SAM; @Cheetham:2016) are not yet supported, but are planned to be included in future updates.

Unlike the ESO archive, spherical consolidates observational metadata, target star properties, and observing conditions into one table, simplifying dataset selection. It integrates an automated *end-to-end data reduction pipeline* for SPHERE’s IFS, enabling spectral cube extraction, calibration, and exoplanet characterization. For IRDIS, spherical supports downloading DBI and polarimetric sequences. By providing an intuitive path from archival data to calibrated products, spherical maximizes SPHERE's scientific yield.

The SPHERE instrument will soon be equipped with a second-stage adaptive optics system (SAXO+; @Stadler:2022) as part of the SPHERE+ upgrade (@Boccaletti:2020; @Boccaletti:2022). This enhancement ensures that SPHERE remains a relevant scientific instrument in the coming years and serves as a pathfinder for the future Planetary Camera and Spectrograph (PCS; @Kasper:2021) on the Extremely Large Telescope (ELT).

---

### Statement of Need

ESO's VLT/SPHERE archive hosts the largest collection of high-contrast imaging data worldwide, but accessing and processing these datasets is complex due to fragmented metadata, manual retrieval, and multiple processing pipelines. Current tools include:
 
1. [High Contrast Data Center (DC)](https://hc-dc.cnrs.fr/) [@Delorme:2017]: Java-based platform providing access to reduced datasets using ESO's pipeline and various post-processing algorithms, but lacking efficient data discovery and easy Python interface.

2. [vlt-sphere](https://github.com/avigan/SPHERE) [@Vigan:2020]: Python wrapper around ESO's Data Reduction and Handling (DRH) pipeline, which automates processing of user-downloaded raw data but does not offer integrated post-processing or structured dataset exploration.

3. [IRDAP](https://irdap.readthedocs.io/en/latest/) [@Holstein:2020ascl]: Python pipeline tailored specifically to IRDIS polarimetric observations, offering automated pre- and post-processing, but requiring manual selection and downloading of raw data and calibrations.

In contrast, spherical provides an integrated solution: a searchable database, automated data retrieval, integrated IFS preprocessing (CHARIS) and post-processing (TRAP) pipelines, and compatibility with alternative post-processing tools (e.g., [VIP](https://vip.readthedocs.io/en/latest/) (@Gonzales:2017; @Christiaens:2023), [pyKLIP](https://pyklip.readthedocs.io/en/latest/) (@Wang:2015ascl), IRDAP).

---

### Database Structure and Pipeline Automation

The spherical database, hosted on [Zenodo](https://doi.org/10.5281/zenodo.15147730), is updated regularly. Users typically begin at Step 4:

1. **Database Generation**: spherical compiles header information from ESO archive files into a regularly updated database.
2. **Cross-Matching with SIMBAD**: Observations are cross-matched with Gaia catalog objects, considering stellar magnitude and proximity.
3. **Observation Table Construction**: Produces a structured table summarizing stellar properties, observing conditions, and metadata.
4. **Data Retrieval**: Enables automated filtering, downloading, and sorting of raw datasets.
5. **IFS Data Reduction**: Performs spectral cube extraction and calibration automatically using CHARIS pipeline [@Samland:2022].
6. **Post-Processing**: Integrates with TRAP [@Samland:2021] for automated exoplanet detection and spectral extraction.

### Scientific Use

This package provides the basis for efficient use of SPHERE data. It provides the foundation for the analysis of large number of exoplanet spectra, detection maps for population statistics of arbitrary samples, or pushing towards improved detection limits for lower mass objects using new methods.

### Future Work

The package framework is extensible. Currently, some SPHERE observing modes are not yet included in the database: Sparse Aperture Masking (SAM; @Cheetham:2016) for all instruments, the long-slit spectroscopy mode for IRDIS, and ZIMPOL. Future updates will progressively expand spherical's functionality, providing support for easily discovering and downloading these datasets. Existing pipelines for these other modes can easily be included for reduction in the spherical workflow.

### Software Attribution

The spherical package makes extensive use of Astropy (@Astropy:2013; @Astropy:2018; @Astropy:2022), astroquery for interacting with the ESO archive and astronomical catalogs [@Ginsburg:2019], NumPy [@Harris:2020] for numerical operations, and pandas [@Pandas] for tabular data handling. For IFS pipeline steps, in addition to the CHARIS pipeline (@Brandt:2017; @Samland:2022), several calibration routines adapted from @Vigan:2020 are employed. These, in turn, include contributions from developers listed in the vlt-sphere [package repository](https://github.com/avigan/SPHERE).

## Acknowledgements

Special thanks to Lukas Welzel for motivating the software’s public release and Elisabeth Matthews for beta testing. Contributors are listed [here](https://github.com/m-samland/spherical/graphs/contributors). We acknowledge ESO for SPHERE datasets and thank developers of CHARIS, TRAP, Astropy, and astroquery.

## References


---
title: 'spherical: A Comprehensive Database and Automated Pipeline for VLT/SPHERE High-Contrast Imaging'
tags:
  - Python
  - astronomy
  - exoplanets
  - protoplanetary disks
  - circumstellar disks
  - high-contrast
  - direct imaging
authors:
  - name: Matthias Samland
    orcid: 0000-0002-0101-8814
    affiliation: 1

affiliations:
  - name: Max-Planck-Institut für Astronomie (MPIA), Königstuhl 17, 69117 Heidelberg, Germany
    index: 1

date: 22 May 2025
bibliography: paper.bib
---

### Summary

The Spectro-Polarimetric High-contrast Exoplanet REsearch instrument (SPHERE, @Beuzit:2019) at the Very Large Telescope (VLT) is among the most advanced ground-based instruments for coronagraphic imaging of exoplanets and circumstellar disks in the optical and near-infrared. Over the last decade, SPHERE has been extraordinarily productive, contributing to over 400 [publications](https://telbib.eso.org/) and leading the largest ground-based survey of exoplanets and circumstellar disks (SHINE, e.g., @Chauvin:2017; @Chomez:2025). It has played a pivotal role in probing the outer regions of planetary systems, revealing both planetary-mass companions and intricate disk structures around nearby stars.

The **spherical** software package provides a structured and searchable database of all SPHERE observations from the ESO archive, offering a streamlined interface to explore, download, and analyze these high-contrast imaging datasets. As of May 2025, it includes ~6000 InfraRed Dual-band Imager and Spectrograph (IRDIS, @Dohlen:2008; @Vigan:2010) observations in the dual-band imaging (DBI) mode, ~1000 in the IRDIS dual-beam polarimetric imaging (DPI, @deBoer:2020; @vanHolstein:2020) mode, and ~4500 with the Integral Field Spectrograph (IFS, @Claudi:2008; @Mesa:2015).

Unlike the ESO archive interface, **spherical** consolidates observational metadata (total exposure time, total parallactic angle rotation, quality flags), target star properties, and observing conditions into a single table, significantly simplifying the identification of optimal datasets for scientific analysis. It also integrates an end-to-end data reduction pipeline specifically designed for SPHERE’s IFS, enabling efficient spectral cube extraction, calibration, post-processing, and exoplanet characterization. For IRDIS, **spherical** supports the discovery and download of dual-band imaging and polarimetric sequences. By offering an intuitive path from archival data to calibrated products, **spherical** empowers researchers to maximize the scientific yield of SPHERE observations.

The SPHERE instrument will soon be equipped with a second-stage adaptive optics system (SAXO+, @Stadler:2022) as part of the SPHERE+ upgrade (@Boccaletti:2020; @Boccaletti:2022). This enhancement will ensure that SPHERE remains a highly relevant scientific instrument in the coming years and also serves as a pathfinder and demonstrator for the future Planetary Camera and Spectrograph (PCS, @Kasper:2021) on the Extremely Large Telescope (ELT).

---

### Statement of Need

The European Southern Observatory's (ESO) VLT/SPHERE archive hosts the largest collection of high-contrast imaging data in the world. Despite its scientific richness, accessing and processing these data remain challenging due to complex retrieval mechanisms and fragmented metadata. In addition to directly downloading raw data and using the ESO pipeline—which requires substantial technical expertise and manual configuration—two primary tools currently exist for accessing and processing SPHERE data:
 
1. [**High Contrast Data Center (DC)**](https://hc-dc.cnrs.fr/) [@Delorme:2017]: A Java-based platform for obtaining reduced data that requires manual user interaction. The DC also offers support for reducing data using the ESO pipeline and a suite of post-processing algorithms. However, it lacks an efficient way to discover or cross-match available data, making it best suited for acquiring a small number of datasets or for assistance in analyzing one’s own observing program.

2. [**vlt-sphere**](https://github.com/avigan/SPHERE) [@Vigan:2020]: A Python wrapper for the official ESO Data Reduction and Handling (DRH, @Pavlov:2008) pipeline, which processes user-downloaded raw data but lacks integration with post-processing tools and a structured overview of available datasets.

**spherical** addresses these limitations by providing a searchable summary of all observation sequences, including detailed data on astrophysical targets, observing modes, instrument configurations, and atmospheric conditions. This database, cross-referenced with astronomical catalogs (e.g., SIMBAD), enables easy exploration of available data and seamless downloading of IRDIS and IFS sequences. In contrast to the **High Contrast Data Center** and the **vlt-sphere** package, **spherical** integrates with the open-source CHARIS pipeline for IFS pre-processing and TRAP for high-contrast exoplanet detection. The pipeline is designed for full automation, supporting efficient spectral cube extraction, pre-processing, and planet detection with minimal manual intervention. Furthermore, its outputs are compatible with alternative post-processing tools like IRDAP (@Holstein:2020ascl) and VIP (@Gonzales:2017; @Christiaens:2023), offering flexibility for different scientific post-processing workflows.

---

### Database Structure and Pipeline Automation

The database generated by **spherical** is hosted on [Zenodo](https://doi.org/10.5281/zenodo.15147730) (DOI: `10.5281/zenodo.15147730`) and can be easily updated with new data. The processing workflow of **spherical** consists of several automated steps:

1. **Database Generation:**

   * **spherical** retrieves header information for all VLT/SPHERE IRDIS and IFS files available in the ESO archive, compiling them into a main "table of files." This database is periodically updated as new observations are added. Detailed examples of the generation process are available in the repository documentation and example notebooks.

2. **Cross-Matching with SIMBAD Catalog:**

   * Using header information, the software cross-matches observation epochs and telescope pointings with the Gaia catalog to identify the observed astrophysical targets. Proper motions are accounted for, and by default, matching is constrained to stellar objects with J-band magnitudes <15 mag and parallaxes within 1 kpc to increase fidelity. These parameters are customizable during database creation. As an extreme adaptive optics instrument, VLT/SPHERE requires bright natural guide stars, which facilitates cross-matching with stellar targets.

3. **Observation Table Construction:**

   * After cross-matching, **spherical** generates a structured "table of observations" that includes stellar properties, observing conditions, setup parameters, metadata, and quality flags for each sequence. This table is accessible for exploration via the provided Jupyter notebooks and is also hosted on [Zenodo](https://doi.org/10.5281/zenodo.15147730).

4. **Data Retrieval:**

   * Users can filter observations by target, observing conditions, mode, or program ID. **spherical** then handles automated download and sorting of raw datasets directly from the ESO archive. The downloader currently supports IFS, IRDIS, and IRDIS polarimetry.

5. **Data Reduction for IFS:**

   * For IFS datasets, **spherical** performs spectral cube extraction using the improved open-source Python-based CHARIS pipeline [@Samland:2022], followed by custom photometric and astrometric calibration adapted from routines used in @Vigan:2020. These reduction steps are fully automated, requiring minimal user input. Complete examples of the IFS reduction workflow are provided in the repository's example scripts.

6. **Post-Processing and Planet Detection:**

   * After calibration, **spherical** integrates with the TRAP algorithm [@Samland:2021] for high-contrast imaging post-processing, automatically detecting point sources and extracting their spectra. Detection is enhanced using spectral templates for L- and T-type companions provided by the **species** package [@Stolker:2020], leveraging the spectral dimension of the IFS data.

This structured workflow greatly simplifies handling of VLT/SPHERE data, allowing researchers to **discover and access** scientifically ready products with minimal overhead. For further technical details and hands-on examples, users are encouraged to consult the repository documentation and example notebooks.

### Future Work

The package framework is extensible. Currently, some SPHERE observing modes are not yet included in the database: Sparse Aperture Masking (SAM, @Cheetham:2016) for all instruments, the long-slit mode for IRDIS [@Vigan:2008], and the optical Zurich Imaging Polarimeter (ZIMPOL, @Schmid:2018). Future updates will progressively expand **spherical**'s functionality, providing support for easily discovering and downloading these datasets.

### Software Attribution

The **spherical** package makes extensive use of **astropy** (@Astropy:2013; @Astropy:2018; @Astropy:2022), **astroquery** for interacting with the ESO archive and astronomical catalogs [@Ginsburg:2019], **NumPy** [@Harris:2020] for numerical operations, and **pandas** [@Pandas] for tabular data handling. For IFS pipeline steps, in addition to the CHARIS pipeline (@Brandt:2017; @Samland:2022), several calibration routines adapted from @Vigan:2020 are employed. These, in turn, include contributions from developers listed in the **vlt-sphere** [package repository](https://github.com/avigan/SPHERE).

## Acknowledgements

I would like to give special thanks to Lukas Welzel for his interest in this software package and for motivating me to refactor it for public use by submitting the first external pull request. Thanks also go to Elisabeth Matthews for beta testing. An up-to-date list of contributors to **spherical** is available [here](https://github.com/m-samland/spherical/graphs/contributors). We acknowledge the European Southern Observatory (ESO) for providing the SPHERE datasets through its archive, and the developers of the CHARIS pipeline and TRAP post-processing algorithm for their foundational contributions. We also thank the developers of **astropy** and **astroquery** for providing the framework that enables seamless interaction with data archives and astronomical catalogs.

## References


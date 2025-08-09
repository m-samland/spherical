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

date: 09 August 2025
bibliography: paper.bib
---

### Summary

The Spectro-Polarimetric High-contrast Exoplanet REsearch instrument (SPHERE; @Beuzit:2019) at the Very Large Telescope (VLT) is a leading facility for coronagraphic imaging of exoplanets and circumstellar disks in the optical and near-infrared. Over the last decade, SPHERE has contributed to hundreds of publications and major legacy surveys (e.g., SHINE; @Chauvin:2017; @Chomez:2025).

`spherical` is both a *curated, searchable database* listing all VLT/SPHERE observations and a *Python-based, automated analysis pipeline* for SPHERE’s *Integral Field Spectrograph (IFS)*. The database—archived on Zenodo and regenerable from the ESO archive—consolidates observational metadata, stellar properties, and observing conditions into a single, analysis-ready table. The pipeline takes users from raw IFS frames to calibrated spectral cubes and exoplanet characterization within a script-driven, configurable workflow.

**What `spherical` can do**
- Search and filter the complete SPHERE observation history by target, stellar properties, observing mode, date, and observing conditions  
- Download selected raw datasets (and associated calibrations) directly from the ESO archive  
- Automatically reduce *IFS* data from raw frames to calibrated spectral cubes (using the adapted CHARIS pipeline)  
- Perform astrometric/photometric calibration and *post-processing with TRAP* for companion detection and spectral extraction  
- Provide *IRDIS* (DBI and DPI) dataset discovery and download for subsequent analysis with community tools

As of May 2025, the database includes approximately *6000* IRDIS dual-band imaging (DBI) observations, *~1000* IRDIS dual-beam polarimetric imaging (DPI; @deBoer:2020; @vanHolstein:2020) observations, and *~4500* IFS observations. Other modes—ZIMPOL (@Schmid:2018), IRDIS long-slit spectroscopy (LSS; @Vigan:2008), and Sparse Aperture Masking (SAM; @Cheetham:2016)—are planned for future releases.

SPHERE is undergoing the *SPHERE+* upgrade (@Boccaletti:2020; @Boccaletti:2022), including a second-stage adaptive optics system (SAXO+; @Stadler:2022), ensuring long-term scientific relevance and providing a pathfinder for the ELT’s Planetary Camera and Spectrograph (PCS; @Kasper:2021).

---

### Statement of Need

The ESO VLT/SPHERE archive is the world’s largest collection of high-contrast imaging data, but end-to-end use can be cumbersome due to fragmented metadata, manual data selection, and multiple unintegrated pipelines.

Existing tools address parts of this workflow:
1. **High Contrast Data Center (DC)** [@Delorme:2017]: Java-based access to reduced datasets using ESO’s pipeline and various post-processing algorithms; limited ability for data discovery and programmatic interaction.  
2. **vlt-sphere** [@Vigan:2020]: Python wrappers around ESO pipeline for user-provided raw data; lacks automatic download, integrated post-processing and a unified, searchable database.  
3. **IRDAP** [@Holstein:2020ascl]: Automated IRDIS polarimetry pipeline; manual dataset discovery and retrieval remain user tasks.

**`spherical`** uniquely combines:  
(1) a *complete, regularly updated* SPHERE observation database; (2) *automated retrieval* of raw data and calibrations; and (3) an *integrated, script-driven IFS pipeline* for reduction, post-processing (TRAP), and spectral characterization—while remaining compatible with alternative tools such as VIP (@Gonzales:2017; @Christiaens:2023), pyKLIP (@Wang:2015ascl), and IRDAP.

**Who is it for?**  
Astronomers working on direct imaging of exoplanets and disks, SPHERE survey teams, and researchers assembling large homogeneous samples for population studies.

---

### Design and Implementation

#### Database generation (maintainer workflow)
1. **Header ingestion** — Parse ESO archive headers for all SPHERE observations.  
2. **Cross-matching** — Associate observations with Gaia catalog sources and target metadata (magnitudes, positions), resolving ambiguities by proximity and brightness.  
3. **Observation table construction** — Produce a structured table summarizing stellar properties, observing modes, exposure times, parallactic angle coverage, and observing conditions.  
4. **Archival release** — Publish compiled tables on Zenodo (DOI: https://doi.org/10.5281/zenodo.15147730) for reproducible access, while also providing scripts to regenerate the tables locally.

#### User workflow (analysis)
1. **Discover** — Query and filter the `spherical` tables to identify suitable sequences (IFS or IRDIS modes).  
2. **Retrieve** — Automatically download raw data and required calibrations from ESO.  
3. **Reduce (IFS)** — Extract spectral cubes with the adapted `CHARIS` pipeline (@Brandt:2017; @Samland:2022).  
4. **Calibrate** — Apply astrometric/photometric calibration using routines adapted from @Vigan:2020.  
5. **Post-process** — Run `TRAP` (@Samland:2021) to generate detection maps, estimate detection limits, and extract companion contrast spectra.  
6. **Analyze** — Export to, or interoperate with, community tools (e.g., VIP, pyKLIP, IRDAP) for further analysis.

> **Usage model:** The IFS pipeline is **script-driven** (Python) rather than a single-click CLI. This design exposes key parameters to ensure transparent, reproducible, and optimal reductions across diverse datasets.

---

### Scientific Use

`spherical` lowers the barrier from raw archive files to science-ready products and supports:
- Construction of homogeneous samples for occurrence rate and population studies  
- Re-analyses of archival data with improved algorithms to push detection limits to lower masses  
- Extraction of high S/N spectra for atmospheric characterization of known companions  
- Efficient survey follow-up by rapidly identifying complementary observations across SPHERE modes

---

### Future Work

`spherical` is designed to be extensible. Future releases will:
- Add database coverage and streamlined retrieval for *ZIMPOL*, *IRDIS LSS*, and *SAM* modes  
- Integrate or wrap community reduction pipelines for these modes within the same script-driven framework  
- Provide a small public test dataset for continuous integration of selected pipeline stages  
- Track SPHERE+ updates and adapt calibration/post-processing steps to evolving instrument performance

---

### Software Attribution

`spherical` relies on `Astropy` (@Astropy:2013; @Astropy:2018; @Astropy:2022), `astroquery` for ESO archive and catalog access (@Ginsburg:2019), `NumPy` (@Harris:2020) for numerical operations, and `pandas` (@Pandas) for tabular data handling.  
IFS reduction uses the adapted `CHARIS` pipeline (@Brandt:2017; @Samland:2022), with calibration routines derived in part from @Vigan:2020 (see the `vlt-sphere` repository for individual contributors). Post-processing employs `TRAP` (@Samland:2021).

---

## Acknowledgements

I thank Lukas Welzel for motivating the public release and Elisabeth Matthews for beta testing. Contributors are listed at the project repository’s [contributors page](https://github.com/m-samland/spherical/graphs/contributors). We acknowledge ESO for SPHERE datasets and thank the developers of `CHARIS`, `TRAP`, `Astropy`, and `astroquery`.

## References

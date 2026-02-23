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

# Summary

The Spectro-Polarimetric High-contrast Exoplanet REsearch instrument (SPHERE; @Beuzit:2019) at the Very Large Telescope (VLT) is a leading facility for coronagraphic imaging of exoplanets and circumstellar disks in the optical and near-infrared. Over the last decade, SPHERE has contributed to hundreds of publications and major legacy surveys (e.g., SHINE; @Chauvin:2017; @Chomez:2025).

`spherical` streamlines the end-to-end analysis of SPHERE data by integrating a curated, searchable database with an automated Python-based pipeline. The software enables users to filter the complete observation history by target properties or observing conditions, automatically download raw datasets from the ESO archive, and reduce Integral Field Spectrograph (IFS) data using the adapted CHARIS pipeline. Beyond basic reduction, the workflow handles astrometric calibration and post-processing via TRAP for companion detection, while also facilitating the discovery and retrieval of IRDIS dual-band and polarimetric imaging data for analysis with community tools.

As of May 2025, the database includes approximately 6000 IRDIS dual-band imaging (DBI) observations, ~1000 IRDIS dual-beam polarimetric imaging (DPI; @deBoer:2020; @vanHolstein:2020) observations, and ~4500 IFS observations. SPHERE is undergoing the *SPHERE+* upgrade [@Boccaletti:2020; @Boccaletti:2022], including a second-stage adaptive optics system (SAXO+; @Stadler:2022), ensuring long-term scientific relevance and providing a pathfinder for the ELT’s Planetary Camera and Spectrograph (PCS; @Kasper:2021).

# Statement of Need

The ESO VLT/SPHERE archive constitutes the world’s largest collection of high-contrast imaging data to detect exoplanets, substellar companions, and circumstellar disks. However, utilizing this rich dataset for end-to-end research remains problematic. Researchers face challenges through need of manual identification and acquisition of available data and the lack of integration between various reduction pipelines. These barriers disproportionately affect researchers attempting to assemble large homogeneous samples for population studies, homogenous extraction of exoplanet spectra, and survey teams requiring rapid follow-up.

# State of the Field

Several existing tools address isolated components of the SPHERE data workflow. The **High Contrast Data Center (DC)** [@Delorme:2017] provides Java-based access to reduce datasets using ESO’s internal pipeline but offers limited capabilities for programmatic interaction or custom batch processing. **vlt-sphere** [@Vigan:2020] offers Python wrappers around the ESO pipeline for user-provided raw data, yet it lacks automated archival download and integrated post-processing features. For polarimetry, **IRDAP** [@Holstein:2020ascl] serves as a robust automated pipeline, though it leaves dataset discovery and retrieval as manual user tasks.

`spherical` was build to fill the gap that a missing high level overview of what data SPHERE has taken over its lifetime leaves. It creates a unified workflow for identification of data, download, reduction and post-processing (via TRAP). From identification of observations of interest, all the way to the identification of exoplanets and extraction of their spectra `spherical` is an end-to-end farmework. `spherical` can also be used as infrastructure to download and pre-process data for alternative post-processing ecosystems such as VIP [@Gonzales:2017; @Christiaens:2023], pyKLIP [@Wang:2015ascl], and IRDAP.

# Software Design

The architecture of `spherical` is designed as a high-level abstraction layer over the ESO raw archive, providing a systematic interface that currently does not exist for high-contrast imaging data. A central design choice was the decoupling of metadata curation from data reduction. This is achieved through a multi-stage process that ingests science headers from the ESO archive and cross-matches them with the Gaia catalog based on telescope pointings. To handle the inherent noise in archival metadata, the system employs a proximity, brightness, and object type-based resolution logic to resolve target ambiguities, accounting for epoch of observation of proper motion of stellar targets, ensuring the resulting database is both clean and astronomically accurate.

The core package produces a structured, searchable observation table—also available pre-computed on Zenodo ([DOI: 10.5281/zenodo.15147730](https://doi.org/10.5281/zenodo.15147730))—which aggregates information such as observing mode, total exposure time, parallactic angle coverage, and atmospheric conditions. By choosing a local, curated table over live archive queries, `spherical` enables rapid, complex filtering of the entire instrument history that would otherwise be difficult. The table can be generated from scratch using the `spherical`, but it is recommended to use the pre-computed table as an input, in which case it can be augmented with any new data taken since the last release.

Technically, `spherical` adopts a wrapper-based architecture. Instead of reimplementing reduction algorithms, it provides a Pythonic interface to the @Samland:2022 IFS instrument pipeline. This design trade-off prioritizes scientific continuity and maintainability; by "gluing" the complex CHARIS instrumend-based SPHERE IFS pipeline and TRAP high-contrast imaging post-processing workflows into an object-oriented framework, `spherical` transforms specialized code into a user-friendly end-to-end workflow without sacrificing the precision of the original algorithms. This hybrid approach allows the software to act as a bridge between archival discovery and specialized community tools. Likewise, `spherical` can be used as an initial stage for post-processing frameworks like like VIP and IRDAP.

`spherical` relies purely on open-source software: `Astropy` [@Astropy:2013; @Astropy:2018; @Astropy:2022], `astroquery` for ESO archive and catalog access [@Ginsburg:2019], `NumPy` [@Harris:2020] for numerical operations, and `pandas` [@Pandas] for tabular data handling. The IFS reduction uses the adapted `CHARIS` pipeline [@Brandt:2017; @Samland:2022], with calibration routines derived in part from @Vigan:2020 (see the `vlt-sphere` repository for individual contributors). Post-processing employs `TRAP` [@Samland:2021].

# Research Impact Statement

`spherical` has been used in several recently published peer-reviewed studies to analyse SPHERE IFS data [@Franson:2023; @Hammond:2025; @Stolker:2025], improving the extraction of spectra of exoplanetary atmospheres. At the time these works were published, the code had not yet been citable via an existing DOI. Since the database of existing observations has been made available on Zenodo, it has been downloaded more than 400 times. New exoplanet candidates identified through `spherical` have been accepted for follow-up observation in an ESO P117 proposal (117.2A06.001), awarding 28.5 hours---more than 60\% of all time awarded in P117 to SPHERE observations.

The research impact of `spherical` is centered on enabling large-scale, reproducible science that was previously inhibited by the heterogeneous nature of the SPHERE archive. By lowering the technical barrier from raw archival to science-ready data products, the software facilitates the construction of homogeneous samples essential for statistically significant exoplanet occurrence rate and population studies. This systematic approach allows researchers to apply consistent algorithms across the entire instrument lifetime.

Furthermore, the integration of high-performance post-processing tools like TRAP within an automated framework enables the "mining" of archival data for low-mass companions that were missed by earlier, less sensitive reduction techniques. This capability is particularly timely as the astronomical community prepares for the SPHERE+ upgrade and future ELT-era observations; `spherical` provides the necessary infrastructure to benchmark current performance and develop the automated workflows required to handle the next generation of high-contrast imaging data. By shifting the focus from manual data curation to atmospheric characterization and spectral extraction, the package accelerates the transition from exoplanet discovery to detailed physical understanding.

# Future Work

`spherical` is built with extensibility as a core tenet. Future development will focus on expanding database coverage and streamlining data retrieval for additional observing modes, specifically ZIMPOL, IRDIS LSS, and SAM, while integrating their respective community reduction pipelines into the established script-driven framework. To enhance software reliability and facilitate community contributions, we intend to provide a public test dataset for the continuous integration of selected pipeline stages. Furthermore, the package will evolve alongside the SPHERE+ upgrade; calibration and post-processing modules will be adapted to account for the instrument's enhanced performance and the new capabilities of the SAXO+ adaptive optics system.

# AI Usage Disclosure

The authors acknowledge the use of generative AI (Large Language Models) in the development of `spherical` and the preparation of this manuscript. Specifically, AI tools were utilized to generate source code docstrings, draft the project README and release notes, and assist in refactoring existing code to improve maintainability. During the writing process, generative AI was used for language editing and to refine the manuscript's conciseness. 

The authors certify that generative AI was not involved in the original conceptual design, the software architecture, or the underlying scientific methodology. All AI-generated outputs, including code and text, were manually reviewed, verified, and edited by the authors to ensure technical accuracy and integrity.

## Acknowledgements

I thank Lukas Welzel for motivating the public release and Elisabeth Matthews for beta testing. Contributors are listed at the project repository’s [contributors page](https://github.com/m-samland/spherical/graphs/contributors). We acknowledge ESO for SPHERE datasets and thank the developers of `CHARIS`, `TRAP`, `Astropy`, and `astroquery`.

## References

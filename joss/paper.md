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
    orcid: 0000-0001-9992-4067
    affiliation: 1

affiliations:
  - name: Max-Planck-Institut für Astronomie (MPIA), Königstuhl 17, 69117 Heidelberg, Germany
    index: 1

date: 11 November 2026
bibliography: paper.bib
---

# Summary

The Spectro-Polarimetric High-contrast Exoplanet REsearch instrument (SPHERE; @Beuzit:2019) at the Very Large Telescope (VLT) is one of the main facilities for coronagraphic imaging of exoplanets and circumstellar disks in the optical and near-infrared. Over the last decade, SPHERE has contributed to hundreds of publications and major legacy surveys (e.g., the SpHere INfrared survey for Exoplanets, SHINE; @Chauvin:2017; @Chomez:2025).

`spherical` combines a curated, searchable database of the complete SPHERE observation history with automated reduction pipelines for two of its subsystems. Users can filter observations by target star properties, observing mode, or observing conditions, download the selected raw data and calibrations from the European Southern Observatory (ESO) archive, and reduce them end to end. Integral Field Spectrograph (IFS; @Claudi:2008) data are extracted into spectral cubes by a pipeline adapted from the Coronagraphic High Angular Resolution Imaging Spectrograph (CHARIS; @Brandt:2017; @Samland:2022). Data from the Infra-Red Dual-band Imager and Spectrograph (IRDIS; @Dohlen:2008) taken in dual-band (DBI) or broadband imaging are calibrated and pre-processed by a second pipeline that follows the same steps. Both feed post-processing with TRAP (Temporal Reference Analysis of Planets; @Samland:2021), a detection algorithm that identifies companions by modeling the temporal systematics of the starlight instead of subtracting a reference image, and recovers their astrometry and contrast spectra including their respective covariances.

The v3.2.0 database release [@Samland:2026db] covers observations taken between 2014 May and 2026 October. It lists 4384 IFS sequences and 5585 IRDIS sequences, most of the latter in dual-band or broadband imaging. A further 1155 IRDIS sequences use dual-beam polarimetric imaging (DPI; @deBoer:2020; @vanHolstein:2020), and 355 sequences from both subsystems use sparse aperture masking (SAM; @Cheetham:2016). Another 393 sequences of solar-system bodies, galaxies, and other non-stellar targets are kept in separate tables. DPI and SAM data are covered for discovery and download only. Their reduction is left to community tools. SPHERE is undergoing the *SPHERE+* upgrade [@Boccaletti:2020; @Boccaletti:2022], including a second-stage adaptive optics system (SAXO+; @Stadler:2022), which extends its scientific lifetime and makes it a pathfinder for the Planetary Camera and Spectrograph (PCS; @Kasper:2021) of the Extremely Large Telescope (ELT).

# Statement of Need

The ESO VLT/SPHERE archive is the world's largest collection of high-contrast imaging data for detecting exoplanets, substellar companions, and circumstellar disks. Using it end to end remains difficult. The observation history is not available in a form that can be queried by scientific criteria, only by header keywords such as target name or coordinates. Raw data and their calibrations must be queried and retrieved by hand. These barriers are most challenging for researchers assembling large homogeneous samples for population studies or for uniform extraction of exoplanet spectra, and for survey teams needing rapid follow-up. `spherical` is written for observers working on direct imaging of exoplanets and disks, SPHERE survey teams, and archival researchers.

# State of the Field

Several existing tools address isolated components of the SPHERE data workflow. The **High Contrast Data Center (DC)** [@Delorme:2017] reduces datasets with the ESO pipeline recipes behind a Java-based interface, but offers limited capabilities for programmatic interaction or custom batch processing. **vlt-sphere** [@Vigan:2020] drives the same recipes through `esorex` for user-provided raw data, and therefore requires a local ESO pipeline installation. It lacks automated archival download and integrated post-processing. For polarimetry, **IRDAP** [@Holstein:2020ascl] reduces the data automatically, but the user has to find and retrieve the raw data.

None of them gives an overview of what SPHERE has observed over its lifetime. `spherical` provides that overview and connects it to one workflow, from the selection of observations through download and reduction to the detection of companions and the extraction of their spectra with TRAP. Its retrieval and pre-processing stages can also feed other post-processing packages such as VIP [@Gonzales:2017; @Christiaens:2023], pyKLIP [@Wang:2015ascl], and IRDAP.

# Software Design

`spherical` is built as a layer over the ESO raw archive that keeps metadata curation separate from data reduction. Science headers ingested from the ESO archive are cross-matched on telescope pointing against SIMBAD [@Wenger:2000] and Gaia DR3 [@GaiaDR3:2023]. Because archival metadata is noisy, target ambiguities are resolved by proximity, brightness, and object type, with stellar proper motion propagated to the epoch of observation. Each sequence is assigned to a single host star, and other catalogued stars in the field are listed alongside it.

The cross-match produces a structured, searchable observation table, available pre-computed on Zenodo ([DOI: 10.5281/zenodo.15147730](https://doi.org/10.5281/zenodo.15147730)). It aggregates observing mode, total exposure time, parallactic angle coverage, and atmospheric conditions. Accompanying target tables carry Gaia DR3 astrophysical parameters and, via a cross-match against MOCAdb [@Gagne:2026], young-association membership, BANYAN $\Sigma$ membership probabilities, and adopted ages. With these columns, samples can be selected by stellar age and association membership as well as by pointing, conditions, and stellar parameters. Choosing a local, curated table over live archive queries makes complex filtering of the entire instrument history fast. The tables can be built from scratch, but the recommended entry point is the published set, which a single command extends with data taken since the last release. Every build records its provenance, both in a separate file and in FITS header keywords. The record includes the `spherical` version, the ESO query date and coverage range, the Gaia data release, the status of each enrichment source, and the build parameters.

`spherical` wraps the existing reduction algorithms. It provides a Python interface to the IFS cube extraction of @Samland:2022 and to TRAP and combines both in one object-oriented framework. Code written for single datasets thereby runs over hundreds of them without changes to the algorithms or their published behavior. This keeps results comparable with earlier work and limits the code that has to be maintained. The reduction does not depend on ESO's pipeline software. Calibration, extraction, and pre-processing are implemented in Python. The full chain installs with `pip` without `esorex` or the ESO pipeline kits, and every step can be inspected and modified in one language.

The IRDIS imaging pipeline, introduced in v3.0.0, follows the same steps as the IFS pipeline. It builds master calibrations for each observation and pre-processes the data into calibrated cubes with analytic inverse-variance maps. The star center is measured on the satellite-spot (waffle) frames, and for dithered sequences the offsets are propagated through the dithering header keywords. Photometric calibration uses the unocculted point spread function (PSF) without saturated flux frames, and TRAP post-processing matches spectral templates across the two channels. Both subsystems share one configuration model and by default resume an interrupted campaign where it stopped. The pipelines are validated against a benchmark dataset, the simultaneous IRDIS and IFS observation of 51 Eridani b from 2015 September [@Samland:2017]. The astrometry and photometry of the planet recovered from both data sets agree within the uncertainties with the published values [@Samland:2017; @Maire:2019], and regression tests pin both results to frozen baselines. The [documentation](https://spherical-hci.readthedocs.io) presents both reductions as end-to-end tutorials.

`spherical` builds only on community Python packages. It uses `Astropy` [@Astropy:2013; @Astropy:2018; @Astropy:2022], `astroquery` [@Ginsburg:2019] for access to the ESO archive and to catalogs, `NumPy` [@Harris:2020] for numerical operations, and `pandas` [@Pandas] for tables. The IFS reduction uses the adapted CHARIS pipeline [@Brandt:2017; @Samland:2022], with calibration routines derived in part from @Vigan:2020 (see the `vlt-sphere` repository for individual contributors). Stellar effective temperatures for TRAP's spectral templates come from Gaia DR3 where available and otherwise from a tabulation of @Pecaut:2013. Post-processing employs TRAP [@Samland:2021].

# Research Impact Statement

Before this publication, `spherical` was used to extract spectra of exoplanet atmospheres from SPHERE IFS data in several peer-reviewed studies [@Franson:2023; @Hammond:2025; @Stolker:2025]. At the time these works were published, the code was not yet citable via an existing DOI. Since the database of existing observations was made available on Zenodo, it has been downloaded more than 900 times. New exoplanet candidates identified with `spherical` have been accepted for follow-up observations in ESO period P117 (programme 117.2A06.001). The programme received 28.5 hours, more than 60\% of the time allocated to SPHERE in P117.

Reducing the effort needed to go from raw archival data to science-ready products makes it practical to assemble the homogeneous samples that exoplanet occurrence-rate and population studies require, and to reprocess the full twelve-year archive with one algorithm in search of companions missed by earlier, less sensitive techniques. Because database releases are versioned and record their provenance, an analysis can name the database state it used, and others can retrieve that state and repeat the selection.

# Future Work

Database coverage and retrieval will be extended to the remaining observing modes, the Zurich Imaging Polarimeter (ZIMPOL; @Schmid:2018) and IRDIS long-slit spectroscopy [@Vigan:2008], with the respective community reduction pipelines integrated into the same script-driven framework. The package will also follow the SPHERE+ upgrade, with calibration and post-processing adapted to the improved instrument and to SAXO+.

# AI Usage Disclosure

Generative AI was used in the development of `spherical` and in the preparation of this manuscript. The system used was Claude Code (Anthropic), with Claude Sonnet and Claude Opus models of the 4.x and 5 model families, over the period from mid-2025 to the v3.2.0 release in October 2026.

Within the software, AI assistance went mainly into the infrastructure around the scientific algorithms. This covers pipeline orchestration and step sequencing, structured logging and crash reports, the typed configuration dataclasses, the console entry points, and the workflow that builds and updates the database and records its provenance. AI was also used for the test suite and for docstrings, README, and changelog text. In preparing this manuscript, generative AI was used for language editing and to improve conciseness, not to draft its scientific content.

AI was not used for the scientific methodology. The spectral extraction [@Samland:2022] and post-processing [@Samland:2021] algorithms are pre-existing published work and are wrapped unchanged. The calibration methodology, the target-resolution logic that disambiguates archival pointings, the database schema, the decision to wrap the existing algorithms, and the validation strategy are the author's own.

The author reviewed, edited, and validated all AI-assisted output, and made the primary architectural and scientific design decisions. Correctness was tested on scientific results, through the 51 Eridani b benchmark and the regression tests described above, in addition to review of the code.

## Acknowledgements

I thank Lukas Welzel for motivating the public release, Elisabeth Matthews for beta testing, and Tomas Stolker for testing and contributing bug fixes. Contributors are listed on the project repository's [contributors page](https://github.com/m-samland/spherical/graphs/contributors). I acknowledge ESO for the SPHERE data and thank the developers of CHARIS, TRAP, `Astropy`, and `astroquery`.

## References

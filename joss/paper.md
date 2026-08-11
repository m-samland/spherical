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

date: 11 August 2026
bibliography: paper.bib
---

# Summary

The Spectro-Polarimetric High-contrast Exoplanet REsearch instrument (SPHERE; @Beuzit:2019) at the Very Large Telescope (VLT) is a leading facility for coronagraphic imaging of exoplanets and circumstellar disks in the optical and near-infrared. Over the last decade, SPHERE has contributed to hundreds of publications and major legacy surveys (e.g., the SpHere INfrared survey for Exoplanets, SHINE; @Chauvin:2017; @Chomez:2025).

`spherical` combines a curated, searchable database of the complete SPHERE observation history with automated reduction pipelines for two of its subsystems. Users filter observations by target properties, observing mode, or observing conditions, download the selected raw data and calibrations from the European Southern Observatory (ESO) archive, and reduce them end to end. Integral Field Spectrograph (IFS; @Claudi:2008) data are extracted into spectral cubes by a pipeline adapted from the Coronagraphic High Angular Resolution Imaging Spectrograph (CHARIS; @Brandt:2017; @Samland:2022); data from the Infra-Red Dual-band Imager and Spectrograph (IRDIS; @Dohlen:2008) in dual-band imaging (DBI) mode are calibrated and pre-processed by a pipeline that mirrors the IFS workflow step for step. Both feed post-processing with TRAP (Temporal Reference Analysis of Planets; @Samland:2021), a detection algorithm that models the temporal systematics of the residual starlight instead of subtracting a reference image, and recovers the astrometry and contrast spectra of detected companions.

As of the v3.0.0 release described here, the database covers observations taken between 2014 May and 2026 August: 6101 IRDIS DBI sequences, 4733 IFS sequences, 1205 IRDIS dual-beam polarimetric imaging (DPI; @deBoer:2020; @vanHolstein:2020) sequences, and 355 sparse aperture masking (SAM; @Cheetham:2016) sequences across both subsystems. DPI and SAM data are covered for discovery and download only; their reduction is left to community tools. SPHERE is undergoing the *SPHERE+* upgrade [@Boccaletti:2020; @Boccaletti:2022], including a second-stage adaptive optics system (SAXO+; @Stadler:2022), which extends its scientific lifetime and makes it a pathfinder for the Planetary Camera and Spectrograph (PCS; @Kasper:2021) of the Extremely Large Telescope (ELT).

# Statement of Need

The ESO VLT/SPHERE archive is the world's largest collection of high-contrast imaging data for detecting exoplanets, substellar companions, and circumstellar disks. Using it end to end remains difficult. The observation history is not available in a form that can be queried by scientific criteria; raw data and their calibrations must be identified and retrieved by hand; and the available reduction pipelines are integrated with neither step. These barriers fall hardest on researchers assembling large homogeneous samples for population studies or for uniform extraction of exoplanet spectra, and on survey teams needing rapid follow-up. `spherical` is written for those groups: observers working on direct imaging of exoplanets and disks, SPHERE survey teams, and archival researchers.

# State of the Field

Several existing tools address isolated components of the SPHERE data workflow. The **High Contrast Data Center (DC)** [@Delorme:2017] provides Java-based access to reduce datasets using ESO's internal pipeline but offers limited capabilities for programmatic interaction or custom batch processing. **vlt-sphere** [@Vigan:2020] offers Python wrappers around the ESO pipeline for user-provided raw data, yet it lacks automated archival download and integrated post-processing features. For polarimetry, **IRDAP** [@Holstein:2020ascl] is a well-established automated pipeline, though it leaves dataset discovery and retrieval as manual user tasks.

What none of them provides is a high-level view of what SPHERE has observed over its lifetime. `spherical` supplies that view and connects it to a single workflow covering identification, download, reduction, and post-processing with TRAP. From selecting observations of interest through to detecting companions and extracting their spectra, it is an end-to-end framework. It can equally serve as the retrieval and pre-processing stage for alternative post-processing ecosystems such as VIP [@Gonzales:2017; @Christiaens:2023], pyKLIP [@Wang:2015ascl], and IRDAP.

# Software Design

The architecture of `spherical` is a high-level abstraction layer over the ESO raw archive, providing a systematic interface that does not currently exist for high-contrast imaging data. A central design choice was to decouple metadata curation from data reduction. Science headers ingested from the ESO archive are cross-matched on telescope pointing against SIMBAD [@Wenger:2000] and Gaia DR3 [@GaiaDR3:2023]. Because archival metadata is noisy, target ambiguities are resolved by proximity, brightness, and object type, with stellar proper motion propagated to the epoch of observation.

The result is a structured, searchable observation table, also available pre-computed on Zenodo ([DOI: 10.5281/zenodo.15147730](https://doi.org/10.5281/zenodo.15147730)), which aggregates observing mode, total exposure time, parallactic angle coverage, and atmospheric conditions. Accompanying target tables carry Gaia DR3 astrophysical parameters and, via a cross-match against MOCAdb [@Gagne:2026], young-association membership, BANYAN $\Sigma$ membership probabilities, and adopted ages, so samples can be selected on stellar age and association rather than on pointing alone. Choosing a local, curated table over live archive queries makes complex filtering of the entire instrument history fast. The tables can be built from scratch, but the recommended entry point is the published set, which a single command extends with data taken since the last release. Every build records its provenance, both in a sidecar file and in FITS header keywords: the `spherical` version, the ESO query date and coverage range, the Gaia data release, the status of each enrichment source, and the build parameters. A published table set is therefore self-describing, and a result can be traced back to the exact database state that produced it.

Rather than reimplementing reduction algorithms, `spherical` wraps them, providing a Pythonic interface to the spectral extraction of @Samland:2022 and to TRAP. This trade-off prioritizes scientific continuity and maintainability: gluing both into one object-oriented framework turns specialized code into a workflow that can be run over hundreds of datasets, without altering the algorithms or their published behavior.

The IRDIS dual-band pipeline introduced in v3.0.0 mirrors the IFS one step for step: per-observation master calibrations, pre-processing into calibrated cubes with analytic inverse-variance maps, centering from the satellite-spot (waffle) frames with offsets propagated through the dithering header keywords for dithered sequences, photometric calibration from the unocculted point spread function (PSF), and TRAP post-processing with spectral template matching across the two channels. Both subsystems share one configuration model and one entry point, and resume by default, so an interrupted campaign continues where it stopped. The IRDIS reduction was validated end to end against the published photometry and astrometry of 51 Eridani b, and that agreement is pinned by regression tests against frozen baselines.

`spherical` relies purely on open-source software: `Astropy` [@Astropy:2013; @Astropy:2018; @Astropy:2022], `astroquery` for ESO archive and catalog access [@Ginsburg:2019], `NumPy` [@Harris:2020] for numerical operations, and `pandas` [@Pandas] for tabular data handling. The IFS reduction uses the adapted `CHARIS` pipeline [@Brandt:2017; @Samland:2022], with calibration routines derived in part from @Vigan:2020 (see the `vlt-sphere` repository for individual contributors). Stellar effective temperatures for TRAP's spectral templates come from Gaia DR3 where available and otherwise from a tabulation of @Pecaut:2013. Post-processing employs `TRAP` [@Samland:2021].

# Research Impact Statement

`spherical` has been used in several published peer-reviewed studies to analyse SPHERE IFS data [@Franson:2023; @Hammond:2025; @Stolker:2025], improving the extraction of spectra of exoplanetary atmospheres. At the time these works were published, the code was not yet citable via an existing DOI. Since the database of existing observations was made available on Zenodo, it has been downloaded more than 400 times. New exoplanet candidates identified through `spherical` have been accepted for follow-up observation in an ESO P117 proposal (117.2A06.001), awarding 28.5 hours, more than 60\% of all time awarded in P117 to SPHERE observations.

Reducing the effort needed to go from raw archival data to science-ready products makes it practical to assemble the homogeneous samples that exoplanet occurrence-rate and population studies require, and to reprocess the full twelve-year archive with one algorithm in search of companions missed by earlier, less sensitive techniques. Two properties make such work reusable by others. Database releases are versioned and provenance-tracked, so an analysis can name the exact table state it used and a reader can retrieve it. And the reduction has a measured reference point: the IRDIS pipeline reproduces the published photometry and astrometry of 51 Eridani b, with the comparison documented and frozen as a regression test.

# Future Work

Database coverage and retrieval will be extended to the remaining observing modes, the Zurich Imaging Polarimeter (ZIMPOL; @Schmid:2018) and IRDIS long-slit spectroscopy [@Vigan:2008], with the respective community reduction pipelines integrated into the same script-driven framework. Archive-facing tests already replay recorded queries and the 51 Eridani baselines cover post-processing; a public raw test dataset would extend continuous integration to the reduction stages themselves. The package will also track the SPHERE+ upgrade, adapting calibration and post-processing to the instrument's improved performance and to SAXO+.

# AI Usage Disclosure

Generative AI was used in the development of `spherical` and in the preparation of this manuscript. The system used was Claude Code (Anthropic), with Claude Sonnet and Claude Opus models of the 4.x and 5 model families, over the period from mid-2025 to the v3.0.0 release in 2026.

Within the software, AI assistance was concentrated in the infrastructure surrounding the scientific algorithms rather than in the algorithms themselves: pipeline orchestration and step sequencing, the structured logging and crash-report system, the typed configuration dataclasses, the console entry points and the database build, update, and provenance workflow, the test suite, and docstrings, README, and changelog text. In preparing this manuscript, generative AI was used for language editing and to improve conciseness, not to draft its scientific content.

AI was not used for the scientific methodology. The spectral extraction [@Samland:2022] and post-processing [@Samland:2021] algorithms are pre-existing published work and are wrapped unchanged. The calibration methodology, the target-resolution logic that disambiguates archival pointings, the database schema, the decision to wrap rather than reimplement, and the validation strategy are the author's own.

The author reviewed, edited, and validated all AI-assisted output, and made the primary architectural and scientific design decisions. Correctness was established scientifically rather than by code inspection alone: the IRDIS pipeline reproduces the published photometry and astrometry of 51 Eridani b, and both pipelines are covered by regression tests pinned to frozen baselines.

## Acknowledgements

I thank Lukas Welzel for motivating the public release and Elisabeth Matthews for beta testing. Contributors are listed at the project repository's [contributors page](https://github.com/m-samland/spherical/graphs/contributors). We acknowledge ESO for SPHERE datasets and thank the developers of `CHARIS`, `TRAP`, `Astropy`, and `astroquery`.

## References

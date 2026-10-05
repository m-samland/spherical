---
html_theme.sidebar_secondary.remove: true
myst:
  html_meta:
    "description lang=en": "Find SPHERE observations in the ESO archive and reduce IFS and IRDIS data to companion spectra and astrometry."
---

# spherical

```{image} _static/banner.webp
:alt: A figure fishing a planet out of the sky above the four VLT unit telescopes, with the word spherical below
:class: landing-banner dark-light
```

Find SPHERE observations in the archive and reduce IFS and IRDIS data to companion
spectra and astrometry.

```{button-ref} getting-started/installation
:ref-type: doc
:color: primary
:class: landing-button

Install
```
```{button-ref} getting-started/first-query
:ref-type: doc
:color: primary
:outline:
:class: landing-button

First query
```

```{sequence-strip}
:archive: concepts/database
:database: concepts/database
:reduction: concepts/anatomy
:trap: concepts/adi-sdi-trap
:products: concepts/products
```

::::{grid} 1 1 3 3
:gutter: 4
:class-container: landing-columns

:::{grid-item}
**New to spherical**

Install it, download the observation database and run a first query in
[Getting started](getting-started/index.md). Then read
[How spherical works](concepts/index.md).
:::

:::{grid-item}
**Reducing data**

Start from a reduction template and adjust it with the
[configuration reference](reference/configuration.md) and the
[TRAP settings](reference/trap-settings.md). The [How-to guides](how-to/index.md)
cover surveys, re-runs and long runs on a server.
:::

:::{grid-item}
**Looking something up**

Steps, commands, conventions and the Python API are in the
[Reference](reference/index.md).
:::
::::

## What spherical is and is not

spherical has two halves. The first is a database of every SPHERE
{term}`Observation sequence` in the ESO archive, built from the file headers and
matched to Gaia DR3 and the MOCA database of young stars. It holds metadata, such as
the target, observing mode, conditions and quality flags, but no reduced data.

The second is a reduction pipeline for {term}`IFS` data and for {term}`IRDIS`
dual-band imaging. It downloads a sequence, calibrates it, runs {term}`TRAP` and
measures the spectra and positions of companions. You run it from a Python script
with `execute_targets`; there is no command-line tool for reductions. IRDIS
polarimetry and sparse aperture masking are in the database for finding and
downloading data, but the pipeline does not reduce them. Single-channel IRDIS modes
run but are untested.

## Citing

If spherical supports your research, cite
[Samland (2025)](https://arxiv.org/abs/2509.08044) and the papers listed on
[Citing](project/citing.md).

```{toctree}
:hidden:

getting-started/index
concepts/index
how-to/index
reference/index
project/index
```

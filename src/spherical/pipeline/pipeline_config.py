from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

# -------- helpers -----------------------------------------------------------

def _to_dict(maybe_dataclass) -> Dict[str, Any]:
    """Accept either a mapping or one of the *Config dataclasses."""
    if isinstance(maybe_dataclass, dict):
        return maybe_dataclass          # already a dict
    return asdict(maybe_dataclass)       # unwrap dataclass


def _absolute(path: Path | str) -> Path:
    """Expand ``~`` and anchor *path* to the current working directory.

    A relative directory cannot survive the run: TRAP's ``add_default_templates``
    chdirs into the species database directory and never restores the previous
    cwd, so every relative output path written after it (the whole
    ``template_matching/`` tree) silently lands under the species directory
    instead. Anchoring once, at config construction, keeps the paths meaningful
    no matter who changes the cwd later.

    ``os.path.abspath`` rather than ``Path.resolve``: symlinked data roots
    (``/tmp`` on macOS, automounted network shares) must keep the name the user
    gave them, otherwise resumed runs no longer match the paths of earlier ones.
    """
    return Path(os.path.abspath(os.path.expanduser(str(path))))


# -------- calibration -------------------------------------------------------

@dataclass(slots=True)
class CalibrationConfig:
    """IFS only. Settings for the charis wavelength calibration (``reduce_calibration`` step)."""

    #: Passed to charis ``buildcalibrations`` for compatibility, but currently has
    #: no effect: charis always uses its static ``mask.fits``.
    mask: str | None = None
    #: Polynomial order of the lenslet-position fit as a function of wavelength.
    #: ``None`` uses charis's instrument default.
    order: int  | None = None
    #: Also build the oversampled lenslet templates that ``fitshift`` needs.
    #: Keep ``True`` while ``config.extraction.fitshift`` is on.
    upsample: bool = True
    #: CPUs for the calibration. Overwritten from ``config.resources.ncpu_calib``
    #: when a reduction starts; set that, or call ``config.set_ncpu(n)``.
    ncpus: int = 4
    #: Print charis progress messages during the calibration.
    verbose: bool = True

    def merge(self, **kw) -> "CalibrationConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)


# -------- cube extraction ---------------------------------------------------

@dataclass(slots=True)
class ExtractionConfig:
    """IFS only. Settings passed to charis ``getcube`` for the cube extraction.

    Each field is a charis argument of the same name; see the ``getcube``
    docstring in charis for the full description. Defaults differ from charis's
    where noted.
    """

    #: Extract every DIT of a raw file as its own cube. Keep ``True``: the frame
    #: tables have one row per DIT.
    individual_dits: bool = True
    #: Threads per charis process. Keep 1, because spherical already runs one
    #: charis process per frame (``config.resources.ncpu_extract``).
    maxcpus:        int  = 1
    #: Extra noise floor as a fraction of the count rate in the variance model
    #: (charis default 0). 0.05 gives a reduced chi-squared near 1.
    noisefac:      float = 0.05
    #: Detector gain in e-/DN used for the shot-noise variance (charis default 2).
    gain:          float = 1.8
    #: Save the up-the-ramp combined 2-D image as an extra file.
    saveramp:       bool = False
    #: Subtract a background before extraction. Which background is used is
    #: set by ``config.preprocessing.bg_pca`` and ``subtract_coro_from_center``.
    bgsub:          bool = True
    #: Apply the pixel and lenslet flat fields.
    flatfield:      bool = True
    #: Apply the charis bad-pixel mask. Strongly recommended.
    mask:           bool = True
    #: Extraction algorithm: ``"optext"`` (quasi-optimal aperture extraction,
    #: fast) or ``"lstsq"`` (least-squares fit of the lenslet PSFs, slower,
    #: writes residuals). charis default ``"lstsq"``.
    method:         str  = "optext"
    #: Fit a sub-pixel shift between the lenslet templates and the data before
    #: extraction, to follow flexure since the calibration. Always off for FLUX
    #: frames.
    fitshift:       bool = True
    #: Subtract correlated read noise. Not supported for SPHERE: charis turns it
    #: off with a warning, because the ESO pipeline already removes it.
    suppressrn:     bool = False
    #: Minimum percentage of usable pixels for the read-noise estimate of
    #: ``suppressrn``. No effect for SPHERE, where ``suppressrn`` is off.
    minpct:         int  = 70
    #: Run a second least-squares pass that removes lenslet crosstalk.
    #: Roughly doubles the extraction time.
    refine:         bool = True
    #: Fraction of the predicted crosstalk removed in the ``refine`` pass
    #: (charis default 0.8).
    crosstalk_scale:float = 0.98
    #: Apply a spectral (DC) crosstalk correction before extraction.
    dc_xtalk_correction: bool = False
    #: Use a linear rather than logarithmic wavelength grid for ``"optext"``
    #: (charis default ``False``).
    linear_wavelength:   bool = True
    #: Fit an undispersed background in each microspectrum column during a
    #: ``"lstsq"`` extraction (charis default ``True``).
    fitbkgnd:       bool = False
    #: Mark lenslets with anomalously low inverse variance as bad and replace
    #: their flux with a local average (cosmetic).
    smoothandmask:  bool = True
    #: Resample the cube from the hexagonal lenslet grid to a square pixel
    #: grid. Keep ``True``; the later steps expect square pixels.
    resample:       bool = True
    #: Write the 2-D residual image of the fit. Only has an effect with
    #: ``method="lstsq"`` (charis default ``False``).
    saveresid:      bool = True
    #: Verbose logging, for charis and for the reduction log.
    verbose:        bool = True
    #: Spectral resolution of the charis calibration to use. Set per observation
    #: to 55 (YJ) or 35 (H) when the reduction starts, so a value set here is
    #: overwritten.
    R: int | None = None

    def merge(self, **kw) -> "ExtractionConfig":
        return replace(self, **kw)


# -------- generic pre-processing -------------------------------------------

#: The frame types a reduction can extract; ``frame_types_to_extract`` narrows them.
FRAME_TYPES = ("CORO", "CENTER", "FLUX")


def validate_frame_types(frame_types) -> None:
    """Raise if ``frame_types`` is empty or names a type not in :data:`FRAME_TYPES`.

    Case does not matter. Called when a config is built, so a typo such as
    ``"CENTRE"`` fails before any download instead of dropping CENTER silently.
    """
    requested = {str(ft).upper() for ft in frame_types}
    if not requested:
        raise ValueError("frame_types_to_extract is empty.")
    unknown = sorted(requested - set(FRAME_TYPES))
    if unknown:
        raise ValueError(
            f"frame_types_to_extract has unsupported frame types {unknown}; "
            f"choose from {list(FRAME_TYPES)}."
        )


@dataclass(slots=True)
class PreprocConfig:
    """Pre-processing, flux calibration and ESO download settings shared by IFS and IRDIS."""

    #: IFS only. Number of charis extractions run in parallel. Overwritten from
    #: ``config.resources.ncpu_extract`` when a reduction starts; set that, or
    #: call ``config.set_ncpu(n)``.
    ncpu_cubebuilding: int  = 4
    #: IFS only. Let charis model the background of CORO and FLUX frames with
    #: PCA instead of subtracting the observation's BG_SCIENCE frame. Without a
    #: BG_SCIENCE frame the PCA background is used anyway. CENTER frames always
    #: use the PCA background unless ``subtract_coro_from_center`` is set.
    bg_pca:            bool = True
    #: IFS only. Use the nearest CORO frame as the background of each CENTER
    #: frame, which removes the stellar halo around the waffle spots.
    subtract_coro_from_center:  bool = False
    #: Drop the first frame of the first flux block before combining it, when
    #: the block has more than one frame. Guards against settling effects after
    #: an instrument change.
    exclude_first_flux_frame:   bool = True
    #: Drop the first frame of every flux block, the first included, when the
    #: block has more than one frame.
    exclude_first_flux_frame_all: bool = True
    #: How the flux frames of a block are combined: ``"median"`` or ``"mean"``.
    flux_combination_method:    str  = "median"
    #: IRDIS only. Which flux cubes (raw files) calibrate the PSF. ``"auto"``
    #: drops cubes whose PSF core reaches ``flux_saturation_adu`` and keeps all
    #: others; DIT and ND are scaled per frame, so mixed setups combine
    #: correctly. ``"before"`` / ``"after"`` do the same for the cubes on one
    #: side of the science sequence. ``"all"`` keeps every cube unmeasured; an
    #: integer index or an ORIGFILE name keeps exactly that cube.
    flux_cube_selection_irdis:  str | int = "auto"
    #: IFS only. As ``flux_cube_selection_irdis``. Defaults to ``"all"``
    #: because the extracted IFS cube is not in detector ADU, so the saturation
    #: test does not apply.
    flux_cube_selection_ifs:    str | int = "all"
    # Above the 35 000 ADU 1% linearity ceiling in the SPHERE User Manual and
    # equal to the DRH's saturated/unsaturated discriminator
    # (sph_ifs_detector_persistence threshold_upper).
    #: Core peak (ADU) at which a flux cube counts as saturated and is dropped
    #: by ``"auto"`` selection. Above the 35 000 ADU 1% linearity limit in the
    #: SPHERE User Manual and equal to the DRH's saturation threshold.
    flux_saturation_adu:        float = 40000.0
    #: Kept flux cubes whose core peak reaches this level (ADU) are reported as
    #: possibly non-linear. Below the 35 000 ADU 1% linearity limit.
    flux_nonlinearity_adu:      float = 30000.0
    #: CPUs for the waffle-spot centre fit. Overwritten from
    #: ``config.resources.ncpu_center`` when a reduction starts; set that, or
    #: call ``config.set_ncpu(n)``.
    ncpu_find_center: int  = 4
    #: Number of frames, spread over the sequence, that get a diagnostic plot of
    #: the waffle-spot fit. ``None`` plots every frame, ``0`` none. Plotting
    #: takes most of the centre-fitting time; every frame of an IFS sequence is
    #: more than 10 000 pages.
    n_center_plots: int | None = 10
    #: Frame types to reduce, any of ``"FLUX"``, ``"CENTER"``, ``"CORO"`` (case
    #: does not matter). Types left out are not extracted (IFS) or
    #: pre-processed (IRDIS).
    frame_types_to_extract: list[str] = field(default_factory=lambda: ['FLUX', 'CENTER', 'CORO'])

    #: ESO user name for downloading proprietary data. ``None`` downloads public
    #: data anonymously.
    eso_username: str | None = None
    #: Keep the ESO password in the system keyring so you are asked for it only
    #: once.
    store_password: bool = True
    #: Delete the stored ESO password from the keyring after all targets are
    #: reduced.
    delete_password_after_reduction: bool = True

    def __post_init__(self) -> None:
        validate_frame_types(self.frame_types_to_extract)

    def merge(self, **kw) -> "PreprocConfig":
        return replace(self, **kw)

# -------- resources ---------------------------------------------------------

@dataclass(slots=True)
class Resources:
    """CPU budget per stage. ``config.set_ncpu(n)`` sets all of them at once."""

    #: CPUs for the calibration step (IFS wavelength calibration).
    ncpu_calib: int = 4
    #: IFS only. Number of charis extractions run in parallel.
    ncpu_extract: int = 4
    #: CPUs for the waffle-spot centre fit.
    ncpu_center: int = 4
    #: CPUs for TRAP. Reaches TRAP only through
    #: ``config.apply_trap_resources(trap_config)``.
    ncpu_trap: int = 4
    # Distinct from ncpu_extract because IRDIS preprocessing replaces the charis
    # extract path and its cost is driven by bad-pixel density, not extraction.
    #: IRDIS only. Workers for the ``preprocess_irdis`` step (background
    #: subtraction, flat field and bad-pixel correction per frame).
    ncpu_preprocess: int = 4

    @property
    def ncpu(self) -> int | None:
        """Get the master ncpu value if all individual values are the same, otherwise None."""
        if (
            self.ncpu_calib == self.ncpu_extract == self.ncpu_center
            == self.ncpu_trap == self.ncpu_preprocess
        ):
            return self.ncpu_calib
        return None

    @ncpu.setter
    def ncpu(self, value: int):
        """Set all CPU parameters to the same value."""
        self.ncpu_calib = value
        self.ncpu_extract = value
        self.ncpu_center = value
        self.ncpu_trap = value
        self.ncpu_preprocess = value

    def apply(self,
              calib: CalibrationConfig,
              pre: PreprocConfig):
        """Apply CPU resource settings to configuration objects."""
        calib.ncpus           = self.ncpu_calib
        pre.ncpu_cubebuilding = self.ncpu_extract
        pre.ncpu_find_center  = self.ncpu_center

    def merge(self, **kw) -> "Resources":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)

# -------- directory configuration -------------------------------------------

_DIRECTORY_FIELDS = frozenset({"base_path", "raw_directory", "reduction_directory"})


@dataclass(slots=True)
class DirectoryConfig:
    """Configuration for data directories and paths."""
    #: Root of all data. Relative paths and ``~`` are made absolute.
    base_path: Path | str = field(default_factory=lambda: Path.home() / "data/sphere")
    #: Where raw ESO files are downloaded. ``None`` means ``base_path / "data"``.
    raw_directory: Path | str | None = None
    #: Where reduction products are written. ``None`` means
    #: ``base_path / "reduction"``.
    reduction_directory: Path | str | None = None

    def __setattr__(self, name, value):
        """Normalize the three directory fields however they are assigned.

        Both reduction templates configure the layout by assignment on an
        existing config (``config.directories.base_path = ...``), which never
        reaches ``__post_init__``. Normalizing here is what makes the guarantee
        hold for the documented usage and not just for the constructor.

        ``object.__setattr__`` rather than zero-argument ``super()``:
        ``slots=True`` makes ``@dataclass`` build a replacement class, and on
        interpreters without the gh-90562 fix the ``__class__`` cell captured by
        ``super()`` still points at the discarded original, so every
        instantiation raises ``TypeError: obj is not an instance or subtype of
        type``. The backport reached 3.14 and late 3.13 patch releases only;
        3.12.11 and 3.11 still raise, and both are supported here.
        """
        if name in _DIRECTORY_FIELDS and value is not None:
            value = _absolute(value)
        object.__setattr__(self, name, value)

    def __post_init__(self):
        """Set default paths based on base_path if not explicitly provided."""
        # __setattr__ has already absolutized whatever was passed in; only the
        # base_path-derived defaults are still missing.
        if self.raw_directory is None:
            self.raw_directory = self.base_path / "data"

        if self.reduction_directory is None:
            self.reduction_directory = self.base_path / "reduction"

    def merge(self, **kw) -> "DirectoryConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)

    def get_paths_dict(self) -> Dict[str, Path]:
        """Return dictionary of all configured paths."""
        # After __post_init__, all paths are guaranteed to be Path objects
        return {
            'base_path': self.base_path,  # type: ignore
            'raw_directory': self.raw_directory,  # type: ignore
            'reduction_directory': self.reduction_directory,  # type: ignore
        }

# -------- pipeline steps configuration ----------------------------------

@dataclass(slots=True)
class PipelineStepsConfig:
    """Which pipeline steps run, and whether finished steps are recomputed."""

    #: Download the observation's science and calibration frames from the ESO archive.
    download_data: bool = True
    #: IFS only. Build the charis wavelength calibration.
    reduce_calibration: bool = True
    #: IFS only. Extract a spectral cube with charis from every raw frame.
    extract_cubes: bool = True
    #: IFS only. Combine the extracted cubes into one cube per frame type.
    bundle_output: bool = True
    #: Write pipeline version and provenance into the FITS headers of the cubes.
    cube_header_update: bool = True
    #: IRDIS only. Build the master background, flat field and bad-pixel map.
    irdis_calibration: bool = True
    #: IRDIS only. Calibrate the raw frames into cubes and inverse-variance cubes.
    preprocess_irdis: bool = True
    #: IFS only. Also bundle the cubes on the native hexagonal lenslet grid.
    bundle_hexagons: bool = False
    #: IFS only. Also bundle the charis fit residuals (needs ``method="lstsq"``).
    bundle_residuals: bool = False
    #: Compute times, parallactic and derotation angles for every frame.
    compute_frames_info: bool = True
    #: Fit the star position in every frame and wavelength from the waffle spots.
    find_centers: bool = True
    #: Plot how the fitted star position moves through the sequence.
    plot_image_center_evolution: bool = True
    #: Turn the fitted star positions into the ones TRAP uses: IFS fits a
    #: polynomial across wavelength per frame; IRDIS carries the CENTER-frame
    #: positions to every CORO frame with the header offsets (INS1 PAC X/Y), or,
    #: in continuous-waffle sequences, flags outlier fits in time and
    #: interpolates failed ones.
    process_extracted_centers: bool = True
    #: Measure the flux of the waffle spots in the CENTER frames.
    calibrate_spot_photometry: bool = True
    #: Build the flux-calibrated, unsaturated PSF from the FLUX frames.
    calibrate_flux_psf: bool = True
    #: Scale the waffle-spot fluxes to the PSF to track the stellar flux.
    spot_to_flux: bool = True
    #: Write a copy of the science cube with the star on the central pixel, for
    #: classical ADI or PCA. Off by default because it doubles the cube's disk
    #: use; tune it with ``config.alignment``.
    align_frames: bool = False
    #: Run the TRAP reduction. Read by ``run_trap_on_observations``, which you
    #: call after ``execute_targets``.
    run_trap_reduction: bool = True
    #: Detect companions in the TRAP maps and characterise them. Read by
    #: ``run_trap_on_observations``.
    run_trap_detection: bool = True
    #: Recompute steps whose outputs already exist. ``False`` resumes: enabled
    #: steps with outputs on disk are skipped. ``True`` recomputes every enabled
    #: step. A set of step names, e.g. ``{"extract_cubes"}``, recomputes those
    #: steps and every step after them. A leaf step such as ``align_frames``
    #: named in the set reruns only itself.
    force: bool | set[str] = False

    # Class-level list of all IFS pipeline steps (excludes TRAP and overwrite settings)
    _IFS_STEPS = [
        'download_data',
        'reduce_calibration',
        'extract_cubes',
        'bundle_output',
        'cube_header_update',
        'bundle_hexagons',
        'bundle_residuals',
        'compute_frames_info',
        'find_centers',
        'plot_image_center_evolution',
        'process_extracted_centers',
        'calibrate_spot_photometry',
        'calibrate_flux_psf',
        'spot_to_flux',
        'align_frames',
    ]

    # Class-level list of all IRDIS pipeline steps (excludes TRAP)
    _IRDIS_STEPS = [
        'download_data',
        'irdis_calibration',
        'preprocess_irdis',
        'cube_header_update',
        'compute_frames_info',
        'find_centers',
        'plot_image_center_evolution',
        'process_extracted_centers',
        'calibrate_spot_photometry',
        'calibrate_flux_psf',
        'spot_to_flux',
        'align_frames',
    ]

    def merge(self, **kw) -> "PipelineStepsConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)

    def all_steps_disabled(self) -> bool:
        """Check if all pipeline steps are disabled (IFS + IRDIS union)."""
        all_steps = set(self._IFS_STEPS) | set(self._IRDIS_STEPS)
        return not any(getattr(self, step) for step in all_steps)

    def enable_all_ifs_steps(self):
        """Enable all IFS pipeline steps (excludes TRAP and overwrite settings).

        "All" includes the opt-in leaf steps that are off by default, such as
        ``align_frames``. Set those back to False afterwards if the defaults were
        what was wanted.
        """
        for step in self._IFS_STEPS:
            setattr(self, step, True)

    def disable_all_ifs_steps(self):
        """Disable all IFS pipeline steps (excludes TRAP and overwrite settings)."""
        for step in self._IFS_STEPS:
            setattr(self, step, False)

    def enable_all_irdis_steps(self):
        """Enable all IRDIS pipeline steps (excludes TRAP).

        "All" includes the opt-in leaf steps that are off by default, such as
        ``align_frames``. Set those back to False afterwards if the defaults were
        what was wanted.
        """
        for step in self._IRDIS_STEPS:
            setattr(self, step, True)

    def disable_all_irdis_steps(self):
        """Disable all IRDIS pipeline steps (excludes TRAP)."""
        for step in self._IRDIS_STEPS:
            setattr(self, step, False)

# --- Composite reduction config --------------------------------------------

ALIGN_SHIFT_METHODS = ("auto", "fft", "interp", "coarse")

# Padding carried around the frame during the shift and removed afterwards.
# An FFT shift is periodic and would wrap flux from one edge to the other; a
# cubic spline's support reaches 2 px past the border. Shifts are sub-pixel once
# the crop origin puts the star within half a pixel of the centre, so 8 is
# generous for both. Uncropped IRDIS frames need shifts of about 27 px, but
# uncropped frames carry NaN (IRDIS dead bands, IFS field corners), so "auto"
# routes them to the spline, which has no such limit. Only an explicit "fft" or "coarse" can exceed the pad,
# and shift_frame raises then.
DEFAULT_ALIGN_PAD_WIDTH = 8


@dataclass(slots=True)
class AlignmentConfig:
    """Parameters for the optional ``align_frames`` step.

    The aligned cube is a leaf product for external consumers (classical
    ADI/PCA, SDI, inspection). Nothing in the pipeline reads it back.

    The step is gated on a marker file rather than on the aligned cube, so
    deleting the cube does not regenerate it and changing these parameters does
    not re-run it. Use ``config.steps.force = {"align_frames"}`` for either
    (`#177 <https://github.com/m-samland/spherical/issues/177>`_).
    """

    #: How frames are shifted: ``"auto"`` uses an FFT shift on frames without
    #: NaN (cropped IRDIS) and a cubic spline where NaN is present (IFS field
    #: corners, uncropped IRDIS). FFT avoids the spline's photometric smoothing
    #: but is global, so ringing from a filled NaN edge would spread over the
    #: frame. ``"fft"`` and ``"interp"`` force one method; ``"coarse"`` rounds to
    #: a whole-pixel shift without interpolation.
    shift_method: str = "auto"
    #: Padding in pixels around the frame during the shift. Must exceed the
    #: largest shift for ``"fft"`` and ``"coarse"``; the step raises rather than
    #: wrap flux across the frame.
    pad_width: int = DEFAULT_ALIGN_PAD_WIDTH

    def __post_init__(self) -> None:
        if self.shift_method not in ALIGN_SHIFT_METHODS:
            raise ValueError(
                f"shift_method must be one of {ALIGN_SHIFT_METHODS}, got {self.shift_method!r}."
            )
        if self.pad_width < 0:
            raise ValueError(f"pad_width must be >= 0, got {self.pad_width}.")

    def merge(self, **kw) -> "AlignmentConfig":
        return replace(self, **kw)


@dataclass(slots=True)
class IFSReductionConfig:
    """Complete configuration of an IFS reduction, passed to ``execute_targets``."""

    #: Settings for the charis wavelength calibration.
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    #: Settings for the charis cube extraction.
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    #: Pre-processing, flux calibration and ESO download settings.
    preprocessing: PreprocConfig = field(default_factory=PreprocConfig)
    #: Where raw data and reduction products are stored.
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    #: CPU budget per stage; set all at once with ``config.set_ncpu(n)``.
    resources: Resources = field(default_factory=Resources)
    #: Which steps run, and whether finished steps are recomputed.
    steps: PipelineStepsConfig = field(default_factory=PipelineStepsConfig)
    #: Settings for the optional ``align_frames`` step.
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    #: Fill TRAP's stellar parameters for template matching per target from the
    #: database (Gaia DR3, then a spectral-type estimate) instead of the values set
    #: on ``trap_config.detection``.
    use_gaia_stellar_parameters: bool = True

    #: Give TRAP the packaged coronagraph transmission curve, so contrasts close to
    #: the coronagraph are corrected. A table you set on ``trap_config.reduction``
    #: takes precedence.
    apply_coronagraph_transmission: bool = True

    # When True (default), `run_trap_on_observation` loads
    # `converted/{coro,center}_ivar_cube.fits` and passes it to trap as
    # `inverse_variance_full`. Empirically improves detection on IRDIS
    # DBI reference (51 Eri DB_K12 2015-09-24). Set False to skip the disk
    # I/O + memory cost when noise weighting is not wanted.
    #: Pass the inverse-variance cubes to TRAP as noise weights. Improves detection;
    #: turn off to save the disk reads and memory.
    pass_inverse_variance_to_trap: bool = True

    # When True (default) and no calibration bad-pixel map is available, derive
    # TRAP's `bad_pixel_mask_full` from the inverse-variance cube so damaged
    # spaxels are kept out of the regressor pool. IFS never has such a map. Since
    # charis's variance-propagating hexagon-to-square resample (charis issue 013),
    # a lenslet charis flags survives as an exact `ivar == 0` in every square it
    # touches (~6-11% of the illuminated field per channel on 51 Eri OBS_H), so
    # `ivar == 0` is now the primary bad-spaxel test. Requires
    # `pass_inverse_variance_to_trap=True`.
    #: When no calibration bad-pixel map exists, derive TRAP's bad-pixel mask from
    #: the inverse-variance cube, so damaged spaxels stay out of the regressors.
    #: IFS never has such a map; lenslets charis flags arrive as ``ivar == 0``.
    #: Needs ``pass_inverse_variance_to_trap``.
    derive_trap_bad_pixels_from_ivar: bool = True

    # Fraction-of-local-baseline floor for the *secondary* soft-deficit test in
    # `pipeline.ivar_badpixels`. On IFS this is 0.0: bad lenslets are already
    # exact zeros (flagged by the `ivar <= 0` branch above), and the variance-
    # propagating resample imprints a real 3-5x moiré on the ivar, so any
    # positive threshold only flags good moiré troughs (~0.1-0.3% of the
    # illuminated field at 0.2, measured on 51 Eri OBS_H) without catching real
    # defects. Raise only to deliberately re-enable the soft test; see charis
    # issue 013 and `pipeline.ivar_badpixels`.
    #: Extra bad-pixel test: also flag pixels whose inverse variance falls below
    #: this fraction of the local level. 0.0 (off) on IFS, because bad lenslets are
    #: already exact zeros and a positive value only flags good pixels in the
    #: resampling moire pattern.
    ivar_bad_pixel_ratio_threshold: float = 0.0

    # TRAP's bad-pixel mask is 2-D per wavelength, so the per-frame ivar flags
    # are collapsed: a spaxel is masked when bad in more than this fraction of
    # frames. charis's per-frame flagging is overwhelmingly transient (on 51 Eri
    # OBS_H ~82% of the interior is flagged in >=1 of 256 frames but only ~0.1%
    # in all), so `0.5` keeps sigma-clipped cosmics out of the mask and only
    # excludes persistently damaged spaxels from the regressor pool. Lower it to
    # mask spaxels bad in fewer frames (0.25 -> ~2x the mask on 51 Eri, 0.0 ->
    # "bad in any frame" masks ~82%); raise it towards 1.0 to mask only always-
    # bad spaxels. Per-frame zero weighting already neutralises kept spaxels in
    # the reduction area, so this only governs the regressor pool.
    #: TRAP's bad-pixel mask has one map per wavelength, so per-frame flags are
    #: collapsed: a pixel is masked when it is bad in more than this fraction of
    #: frames. 0.5 keeps transient flags such as cosmic rays out of the mask; lower
    #: it to mask more, raise it towards 1.0 to mask only always-bad pixels.
    ivar_bad_pixel_frame_fraction: float = 0.5

    #: For continuous-waffle sequences, pass the measured stellar flux variation
    #: (``spot_amplitude_variation.fits``) to TRAP. Has no effect otherwise.
    pass_amplitude_modulation_to_trap: bool = False

    # When True AND the observation is continuous-waffle, load the CENTER-frame
    # waffle-fit outlier list (`converted/additional_outputs/center_outlier_frames.fits`,
    # written by `process_centers` for that path), union the per-channel
    # outlier indices, and pass the result to trap as `bad_frames`. Since #144
    # and #145 the flagged frames are genuinely rare (a handful in 560 on Beta
    # Pic K12) and their centres are kept as measured, so this is the only place
    # frame rejection happens — it forwards the list downstream so TRAP excludes
    # those frames from the temporal PCA basis.
    # Explicitly gated on continuous-waffle: in non-waffle observations the
    # CORO cube is a separate (usually longer) sequence, so a per-CENTER-frame
    # outlier index has no meaning as a CORO bad_frames index — the flag is
    # ignored with an INFO log even if a stale outliers file exists.
    #: For continuous-waffle sequences, pass the frames whose waffle-spot fit was
    #: an outlier to TRAP as bad frames, so they are left out of its temporal
    #: model. Ignored for other sequences.
    pass_center_outliers_as_bad_frames_to_trap: bool = False

    def as_plain_dicts(self):
        return (
            asdict(self.calibration),
            asdict(self.extraction),
            asdict(self.preprocessing),
            asdict(self.directories),
        )

    def apply_resources(self):
        """Apply resource configuration to all sub-configs."""
        self.resources.apply(self.calibration, self.preprocessing)

    def apply_trap_resources(self, trap_config):
        """Apply CPU resources to TRAP configuration."""
        trap_config.resources.ncpu_reduction = self.resources.ncpu_trap
        trap_config.apply_resources()

    def set_ncpu(self, ncpu: int):
        """Convenience method to set master ncpu and apply it to all configurations."""
        self.resources.ncpu = ncpu
        self.apply_resources()

# Factory method for creating default config
def defaultIFSReduction() -> IFSReductionConfig:
    return IFSReductionConfig()


# -------- IRDIS-specific calibration & preprocess sub-configs --------------

@dataclass(slots=True)
class IRDISCalibrationConfig:
    """IRDIS only. Master-calibration parameters for the IRDIS calibration step.

    Controls the construction of the master background, master flat, and
    bad-pixel map from archive FLAT and BG_SCIENCE frames.
    """
    #: Not used yet.
    combination_method: str = "median"
    #: Flag flat-field pixels deviating from 1.0 by more than this many robust
    #: sigmas as bad.
    flat_badpix_sigma: float = 5.0
    #: Flag background pixels that are hot or noisy by more than this many robust
    #: sigmas as bad.
    background_badpix_sigma: float = 5.0
    #: Flag flat-field pixels with a relative response below this as bad.
    flat_relative_response_min: float = 0.5
    #: Flag flat-field pixels with a relative response above this as bad.
    flat_relative_response_max: float = 1.5
    #: Not used yet. Overwritten from ``config.resources.ncpu_calib`` when a
    #: reduction starts.
    ncpus: int = 4

    def merge(self, **kw) -> "IRDISCalibrationConfig":
        return replace(self, **kw)


@dataclass(slots=True)
class IRDISPreprocessConfig:
    """IRDIS only. IRDIS-detector-specific preprocessing parameters.

    Distinct from the shared ``PreprocConfig`` (which carries ESO download
    settings and shared frame-type controls). Fields here are consumed by
    the ``preprocess_irdis`` step (Phase 4).
    """
    #: Cut the CORO and CENTER frames of each channel to a ``crop_size`` square
    #: around the star, which saves disk space and time in the later steps.
    #: FLUX frames stay full-frame.
    crop: bool = False
    # Must be ODD. TRAP takes the image centre as `yx_dim[0] // 2`; for odd N
    # that integer *is* the array's geometric centre, so TRAP's convention, the
    # geometric centre and the pixel the star sits on are one point. For even N
    # they differ by half a pixel, which FFT rotation/scaling and any symmetry
    # assumption do not tolerate, and an even axis also carries an unpaired
    # Nyquist bin that leaks ringing into a real-valued FFT shift.
    #: Side of the cropped square in pixels. Must be odd, so the star sits on the
    #: central pixel ``N // 2`` that TRAP assumes.
    crop_size: int = 257
    #: Pixel ``(x, y)`` to crop around in both channels. ``None`` crops around
    #: the filter's nominal star position in each channel.
    crop_center: tuple[int, int] | None = None
    #: Replace bad pixels by interpolation from their neighbours.
    fix_badpix: bool = True
    #: Stretch the images to correct the SPHERE anamorphism. Off by default
    #: because TRAP corrects it in its forward model (``trap_config_for_irdis()``
    #: sets ``yx_anamorphism=[1.0062, 1.0]``). If you turn this on, also set
    #: TRAP's ``yx_anamorphism`` to ``[1.0, 1.0]``, or the correction is applied twice.
    correct_anamorphism: bool = False
    #: Stretch factor along y used when ``correct_anamorphism`` is on.
    anamorphism_factor: float = 1.0062
    #: Detector gain in e-/ADU for the analytic inverse-variance map.
    gain: float = 1.75
    #: Read noise in e- for the analytic inverse-variance map.
    read_noise: float = 4.4
    # Conservative radius for the star/PSF exclusion mask in the scaled-background
    # fit. 285 px covers the K-band AO-corrected halo out to where the image is
    # background-dominated (measured on the beta Pic DB_K12 reference set); FLUX
    # frames use a smaller radius because the PSF is compact off the coronagraph.
    #: Radius in pixels of the region around the star left out of the scaled
    #: background fit. 285 px covers the K-band halo out to where the background
    #: dominates.
    star_mask_radius: int = 285
    #: As ``star_mask_radius``, for FLUX frames, whose PSF is compact.
    flux_star_mask_radius: int = 150
    # Per-frame transient sigma-clip threshold (imutils.sigma_filter box=7).
    # DEFAULT DISABLED (0.0). On real IRDIS data the sigma-clip is dominated by
    # AO speckle chatter and waffle-spot residuals, not by actual cosmic-ray
    # transients: at 5σ we measured ~500 flagged pixels/frame-channel vs the
    # expected ~5-20 CR pixels/frame, and the ~110 ms/frame-channel convolution
    # cost is ~25% of the wall time. Downstream operations that consult ivar
    # already handle rare real CRs implicitly (the analytic ivar shrinks at
    # spiky pixels). Turn it back on by setting to e.g. 8.0 if visual streaks
    # in cube medians are a concern. Non-FLUX only; 0.0 means skip entirely.
    #: Sigma threshold for clipping transients (cosmic rays) per frame, in CORO
    #: and CENTER frames only; 0.0 turns it off. Off by default because on real data it mostly flags speckles
    #: and waffle residuals and costs about a quarter of the run time. Try 8.0 if
    #: streaks show up in the cube medians.
    transient_nsigma: float = 0.0

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}.")
        if self.crop_size % 2 == 0:
            raise ValueError(
                f"crop_size must be odd, got {self.crop_size}. "
                f"Use {self.crop_size - 1} or {self.crop_size + 1}. "
                "An odd size makes TRAP's `N // 2` centre coincide with the "
                "array's geometric centre; an even size puts them half a pixel "
                "apart. The value is rejected rather than rounded so the "
                "configured size is always the size that is used."
            )

    def merge(self, **kw) -> "IRDISPreprocessConfig":
        return replace(self, **kw)


# --- IRDIS composite reduction config --------------------------------------

@dataclass(slots=True)
class IRDISReductionConfig:
    """Composite configuration for the IRDIS reduction pipeline."""

    #: Pre-processing, flux calibration and ESO download settings.
    preprocessing: PreprocConfig = field(default_factory=PreprocConfig)
    #: Where raw data and reduction products are stored.
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    #: CPU budget per stage; set all at once with ``config.set_ncpu(n)``.
    resources: Resources = field(default_factory=Resources)
    #: Which steps run, and whether finished steps are recomputed.
    steps: PipelineStepsConfig = field(default_factory=PipelineStepsConfig)
    #: Settings for the optional ``align_frames`` step.
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    #: Settings for the IRDIS master calibrations.
    calibration: IRDISCalibrationConfig = field(default_factory=IRDISCalibrationConfig)
    #: Settings for the IRDIS frame pre-processing.
    irdis_preprocessing: IRDISPreprocessConfig = field(default_factory=IRDISPreprocessConfig)
    #: Same as for IFS: TRAP stellar parameters per target from the database.
    use_gaia_stellar_parameters: bool = True
    #: Same as for IFS: give TRAP the packaged coronagraph transmission curve.
    apply_coronagraph_transmission: bool = True
    #: Same as for IFS: pass the inverse-variance cubes to TRAP as noise weights.
    pass_inverse_variance_to_trap: bool = True
    #: Same as for IFS. On IRDIS the calibration bad-pixel map normally exists and
    #: wins; this only applies when ``badpixel_map.fits`` is missing.
    derive_trap_bad_pixels_from_ivar: bool = True
    #: Same as for IFS, but on by default (0.2): IRDIS inverse variance has no
    #: resampling pattern, so low values do mark bad pixels.
    ivar_bad_pixel_ratio_threshold: float = 0.2
    #: Same as for IFS: mask a pixel bad in more than this fraction of frames.
    ivar_bad_pixel_frame_fraction: float = 0.5
    #: Same as for IFS: pass the stellar flux variation of continuous-waffle sequences.
    pass_amplitude_modulation_to_trap: bool = False
    #: Same as for IFS: pass outlier waffle-fit frames of continuous-waffle sequences.
    pass_center_outliers_as_bad_frames_to_trap: bool = False

    def apply_resources(self) -> None:
        """Copy CPU-budget fields from ``resources`` into sub-configs."""
        self.preprocessing.ncpu_cubebuilding = self.resources.ncpu_extract
        self.preprocessing.ncpu_find_center = self.resources.ncpu_center
        self.calibration.ncpus = self.resources.ncpu_calib

    def apply_trap_resources(self, trap_config) -> None:
        """Apply CPU resources to TRAP configuration.

        `set_ncpu` cannot reach TRAP on its own: the TRAP config is built
        separately by the driver script, so without this call TRAP silently runs
        at its own `TrapReductionConfig.ncpus` default regardless of the budget
        requested here.
        """
        trap_config.resources.ncpu_reduction = self.resources.ncpu_trap
        trap_config.apply_resources()

    def set_ncpu(self, ncpu: int) -> None:
        """Set master CPU budget and apply it to all configurations."""
        self.resources.ncpu = ncpu
        self.apply_resources()


def defaultIRDISReduction() -> IRDISReductionConfig:
    """Return an ``IRDISReductionConfig`` populated with default field values."""
    return IRDISReductionConfig()

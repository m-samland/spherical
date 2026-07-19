from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

# -------- helpers -----------------------------------------------------------

def _to_dict(maybe_dataclass) -> Dict[str, Any]:
    """Accept either a mapping or one of the *Config dataclasses."""
    if isinstance(maybe_dataclass, dict):
        return maybe_dataclass          # already a dict
    return asdict(maybe_dataclass)       # unwrap dataclass


# -------- calibration -------------------------------------------------------

@dataclass(slots=True)
class CalibrationConfig:
    mask: str | None = None          # use standard mask
    order: int  | None = None        # polynomial order
    upsample: bool = True
    ncpus: int = 4
    verbose: bool = True

    def merge(self, **kw) -> "CalibrationConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)


# -------- cube extraction ---------------------------------------------------

@dataclass(slots=True)
class ExtractionConfig:
    individual_dits: bool = True
    maxcpus:        int  = 1  # always 1 inside CHARIS
    noisefac:      float = 0.05
    gain:          float = 1.8
    saveramp:       bool = False
    bgsub:          bool = True
    flatfield:      bool = True
    mask:           bool = True
    method:         str  = "optext"
    fitshift:       bool = True
    suppressrn:     bool = False
    minpct:         int  = 70
    refine:         bool = True
    crosstalk_scale:float = 0.98
    dc_xtalk_correction: bool = False
    linear_wavelength:   bool = True
    fitbkgnd:       bool = False
    smoothandmask:  bool = True
    resample:       bool = True
    saveresid:      bool = True
    verbose:        bool = True
    # dynamic field – filled in by execute_target when YJ/H is known
    R: int | None = None             

    def merge(self, **kw) -> "ExtractionConfig":
        return replace(self, **kw)


# -------- generic pre-processing -------------------------------------------

@dataclass(slots=True)
class PreprocConfig:
    ncpu_cubebuilding: int  = 4
    bg_pca:            bool = True
    subtract_coro_from_center:  bool = False
    exclude_first_flux_frame:   bool = True
    exclude_first_flux_frame_all: bool = True
    flux_combination_method:    str  = "median"
    ncpu_find_center: int  = 4
    frame_types_to_extract: list[str] = field(default_factory=lambda: ['FLUX', 'CENTER', 'CORO'])
    
    # ESO data download settings
    eso_username: str | None = None
    store_password: bool = True # Temporarily store password in keyring
    delete_password_after_reduction: bool = True #Remove password after all reductions are done

    def merge(self, **kw) -> "PreprocConfig":
        return replace(self, **kw)

# -------- resources ---------------------------------------------------------

@dataclass(slots=True)
class Resources:
    ncpu_calib: int = 4
    ncpu_extract: int = 4
    ncpu_center: int = 4
    ncpu_trap: int = 4
    # Worker count for the IRDIS ``preprocess_irdis`` step (parallel per-frame
    # bg subtraction + flat divide + fix_badpix + sigma_filter). Distinct from
    # ncpu_extract because IRDIS preprocessing replaces the charis extract path
    # and its cost profile is bpm-density-driven, not spectral-extraction-driven.
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

@dataclass(slots=True)
class DirectoryConfig:
    """Configuration for data directories and paths."""
    base_path: Path | str = field(default_factory=lambda: Path.home() / "data/sphere")
    raw_directory: Path | str | None = None        # Will default to base_path / "data_test"
    reduction_directory: Path | str | None = None  # Will default to base_path / "reduction_test"

    def __post_init__(self):
        """Set default paths based on base_path if not explicitly provided."""
        # Convert base_path to Path object if it's a string
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        
        # Set defaults for other directories if not provided
        if self.raw_directory is None:
            self.raw_directory = self.base_path / "data"
        elif isinstance(self.raw_directory, str):
            self.raw_directory = Path(self.raw_directory)
            
        if self.reduction_directory is None:
            self.reduction_directory = self.base_path / "reduction"
        elif isinstance(self.reduction_directory, str):
            self.reduction_directory = Path(self.reduction_directory)

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
    """Configuration for which pipeline steps to execute."""
    
    # Data acquisition
    download_data: bool = True
    
    # Core reduction steps
    reduce_calibration: bool = True
    extract_cubes: bool = True
    bundle_output: bool = True
    cube_header_update: bool = True

    # IRDIS-only step flags
    irdis_calibration: bool = True
    preprocess_irdis: bool = True
    
    # Bundle options
    bundle_hexagons: bool = False
    bundle_residuals: bool = False
    
    # Post-processing steps
    compute_frames_info: bool = True
    find_centers: bool = True
    plot_image_center_evolution: bool = True
    process_extracted_centers: bool = True
    calibrate_spot_photometry: bool = True
    calibrate_flux_psf: bool = True
    spot_to_flux: bool = True
    
    # TRAP postprocessing steps
    run_trap_reduction: bool = True
    run_trap_detection: bool = True
    
    # Resume/force control. False = resume (skip enabled steps whose outputs
    # exist); True = redo all enabled steps; a set of step names forces those
    # steps AND every step after them (cascade). Replaces the overwrite_* flags.
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
    ]

    def merge(self, **kw) -> "PipelineStepsConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)

    def all_steps_disabled(self) -> bool:
        """Check if all pipeline steps are disabled (IFS + IRDIS union)."""
        all_steps = set(self._IFS_STEPS) | set(self._IRDIS_STEPS)
        return not any(getattr(self, step) for step in all_steps)

    def enable_all_ifs_steps(self):
        """Enable all IFS pipeline steps (excludes TRAP and overwrite settings)."""
        for step in self._IFS_STEPS:
            setattr(self, step, True)

    def disable_all_ifs_steps(self):
        """Disable all IFS pipeline steps (excludes TRAP and overwrite settings)."""
        for step in self._IFS_STEPS:
            setattr(self, step, False)

    def enable_all_irdis_steps(self):
        """Enable all IRDIS pipeline steps (excludes TRAP)."""
        for step in self._IRDIS_STEPS:
            setattr(self, step, True)

    def disable_all_irdis_steps(self):
        """Disable all IRDIS pipeline steps (excludes TRAP)."""
        for step in self._IRDIS_STEPS:
            setattr(self, step, False)

# --- Composite reduction config --------------------------------------------

@dataclass(slots=True)
class IFSReductionConfig:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    preprocessing: PreprocConfig = field(default_factory=PreprocConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    resources: Resources = field(default_factory=Resources)
    steps: PipelineStepsConfig = field(default_factory=PipelineStepsConfig)

    # When True, TRAP stellar parameters for template matching are populated
    # per observation from the table (Gaia DR3, then spectral-type fallback)
    # instead of using the values configured on trap_config.detection.
    use_gaia_stellar_parameters: bool = True

    # When True, IFS TRAP runs default coronagraph_transmission to the packaged
    # IFS curve (see run_trap._load_coronagraph_transmission). An explicit table
    # set on trap_config.reduction always takes precedence.
    apply_coronagraph_transmission: bool = True

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
    """Master-calibration parameters for the IRDIS calibration step.

    Controls the construction of the master background, master flat, and
    bad-pixel map from archive FLAT and BG_SCIENCE frames.
    """
    combination_method: str = "median"
    flat_badpix_sigma: float = 5.0
    background_badpix_sigma: float = 5.0
    flat_relative_response_min: float = 0.5
    flat_relative_response_max: float = 1.5
    ncpus: int = 4

    def merge(self, **kw) -> "IRDISCalibrationConfig":
        return replace(self, **kw)


@dataclass(slots=True)
class IRDISPreprocessConfig:
    """IRDIS-detector-specific preprocessing parameters.

    Distinct from the shared ``PreprocConfig`` (which carries ESO download
    settings and shared frame-type controls). Fields here are consumed by
    the ``preprocess_irdis`` step (Phase 4).
    """
    crop: bool = False
    crop_size: int = 512
    crop_center: tuple[int, int] | None = None
    fix_badpix: bool = True
    correct_anamorphism: bool = False
    anamorphism_factor: float = 1.0062
    gain: float = 1.75
    read_noise: float = 4.4
    # Conservative radius for the star/PSF exclusion mask in the scaled-background
    # fit. 285 px covers the K-band AO-corrected halo out to where the image is
    # background-dominated (measured on the beta Pic DB_K12 reference set); FLUX
    # frames use a smaller radius because the PSF is compact off the coronagraph.
    star_mask_radius: int = 285
    flux_star_mask_radius: int = 150

    def merge(self, **kw) -> "IRDISPreprocessConfig":
        return replace(self, **kw)


# --- IRDIS composite reduction config --------------------------------------

@dataclass(slots=True)
class IRDISReductionConfig:
    """Composite configuration for the IRDIS reduction pipeline."""

    preprocessing: PreprocConfig = field(default_factory=PreprocConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    resources: Resources = field(default_factory=Resources)
    steps: PipelineStepsConfig = field(default_factory=PipelineStepsConfig)
    calibration: IRDISCalibrationConfig = field(default_factory=IRDISCalibrationConfig)
    irdis_preprocessing: IRDISPreprocessConfig = field(default_factory=IRDISPreprocessConfig)

    use_gaia_stellar_parameters: bool = True
    apply_coronagraph_transmission: bool = True

    def apply_resources(self) -> None:
        """Copy CPU-budget fields from ``resources`` into sub-configs."""
        self.preprocessing.ncpu_cubebuilding = self.resources.ncpu_extract
        self.preprocessing.ncpu_find_center = self.resources.ncpu_center
        self.calibration.ncpus = self.resources.ncpu_calib

    def set_ncpu(self, ncpu: int) -> None:
        """Set master CPU budget and apply it to all configurations."""
        self.resources.ncpu = ncpu
        self.apply_resources()


def defaultIRDISReduction() -> IRDISReductionConfig:
    """Return an ``IRDISReductionConfig`` populated with default field values."""
    return IRDISReductionConfig()

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

    @property
    def ncpu(self) -> int | None:
        """Get the master ncpu value if all individual values are the same, otherwise None."""
        if self.ncpu_calib == self.ncpu_extract == self.ncpu_center == self.ncpu_trap:
            return self.ncpu_calib
        return None

    @ncpu.setter
    def ncpu(self, value: int):
        """Set all CPU parameters to the same value."""
        self.ncpu_calib = value
        self.ncpu_extract = value
        self.ncpu_center = value
        self.ncpu_trap = value

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
    
    # Overwrite settings
    overwrite_calibration: bool = True
    overwrite_bundle: bool = True
    overwrite_preprocessing: bool = True
    overwrite_trap: bool = True
    
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
    
    def merge(self, **kw) -> "PipelineStepsConfig":
        """Return a copy with selected fields overridden."""
        return replace(self, **kw)
    
    def all_steps_disabled(self) -> bool:
        """Check if all pipeline steps are disabled."""
        return not any(getattr(self, step) for step in self._IFS_STEPS)
    
    def enable_all_ifs_steps(self):
        """Enable all IFS pipeline steps (excludes TRAP and overwrite settings)."""
        for step in self._IFS_STEPS:
            setattr(self, step, True)
    
    def disable_all_ifs_steps(self):
        """Disable all IFS pipeline steps (excludes TRAP and overwrite settings)."""
        for step in self._IFS_STEPS:
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

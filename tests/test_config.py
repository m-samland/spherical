"""
Test configuration module for the spherical package.

This module tests the pipeline configuration classes and their basic functionality,
ensuring that configuration objects can be instantiated and modified properly.
"""

from pathlib import Path

import pytest

from spherical.pipeline.pipeline_config import (
    CalibrationConfig,
    DirectoryConfig,
    ExtractionConfig,
    IFSReductionConfig,
    PipelineStepsConfig,
    PreprocConfig,
    Resources,
    defaultIFSReduction,
)


class TestCalibrationConfig:
    """Test the CalibrationConfig class."""

    def test_calibration_config_defaults(self):
        """Test that CalibrationConfig has correct default values."""
        config = CalibrationConfig()
        assert config.mask is None
        assert config.order is None
        assert config.upsample is True
        assert config.ncpus == 4
        assert config.verbose is True

    def test_calibration_config_merge(self):
        """Test that CalibrationConfig.merge() works correctly."""
        config = CalibrationConfig()
        merged = config.merge(ncpus=8, verbose=False)
        
        # Original config should be unchanged
        assert config.ncpus == 4
        assert config.verbose is True
        
        # Merged config should have new values
        assert merged.ncpus == 8
        assert merged.verbose is False
        assert merged.upsample is True  # Unchanged field should remain


class TestExtractionConfig:
    """Test the ExtractionConfig class."""

    def test_extraction_config_defaults(self):
        """Test that ExtractionConfig has correct default values."""
        config = ExtractionConfig()
        assert config.individual_dits is True
        assert config.maxcpus == 1
        assert config.noisefac == 0.05
        assert config.gain == 1.8
        assert config.saveramp is False
        assert config.method == "optext"
        assert config.R is None

    def test_extraction_config_merge(self):
        """Test that ExtractionConfig.merge() works correctly."""
        config = ExtractionConfig()
        merged = config.merge(maxcpus=4, method="boxcar", R=50)
        
        assert config.maxcpus == 1  # Original unchanged
        assert merged.maxcpus == 4  # Merged changed
        assert merged.method == "boxcar"
        assert merged.R == 50
        assert merged.gain == 1.8  # Unchanged field should remain


class TestPreprocConfig:
    """Test the PreprocConfig class."""

    def test_preproc_config_defaults(self):
        """Test that PreprocConfig has correct default values."""
        config = PreprocConfig()
        assert config.ncpu_cubebuilding == 4
        assert config.bg_pca is True
        assert config.subtract_coro_from_center is False
        assert config.flux_combination_method == "median"
        assert config.frame_types_to_extract == ['FLUX', 'CENTER', 'CORO']
        assert config.eso_username is None

    def test_preproc_config_merge(self):
        """Test that PreprocConfig.merge() works correctly."""
        config = PreprocConfig()
        merged = config.merge(
            ncpu_cubebuilding=8,
            eso_username="testuser",
            flux_combination_method="mean"
        )
        
        assert config.ncpu_cubebuilding == 4  # Original unchanged
        assert merged.ncpu_cubebuilding == 8  # Merged changed
        assert merged.eso_username == "testuser"
        assert merged.flux_combination_method == "mean"
        assert merged.bg_pca is True  # Unchanged field


class TestResources:
    """Test the Resources class."""

    def test_resources_defaults(self):
        """Test that Resources has correct default values."""
        resources = Resources()
        assert resources.ncpu_calib == 4
        assert resources.ncpu_extract == 4
        assert resources.ncpu_center == 4
        assert resources.ncpu_trap == 4

    def test_resources_ncpu_property(self):
        """Test the ncpu property getter and setter."""
        resources = Resources()
        
        # Should return 4 when all values are the same
        assert resources.ncpu == 4
        
        # Change one value - should return None
        resources.ncpu_calib = 8
        assert resources.ncpu is None
        
        # Set master ncpu - should update all values
        resources.ncpu = 6
        assert resources.ncpu_calib == 6
        assert resources.ncpu_extract == 6
        assert resources.ncpu_center == 6
        assert resources.ncpu_trap == 6
        assert resources.ncpu == 6

    def test_resources_apply(self):
        """Test that Resources.apply() correctly updates other configs."""
        resources = Resources(ncpu_calib=8, ncpu_extract=6, ncpu_center=2)
        calib_config = CalibrationConfig()
        preproc_config = PreprocConfig()
        
        # Apply resources
        resources.apply(calib_config, preproc_config)
        
        assert calib_config.ncpus == 8
        assert preproc_config.ncpu_cubebuilding == 6
        assert preproc_config.ncpu_find_center == 2

    def test_resources_merge(self):
        """Test that Resources.merge() works correctly."""
        resources = Resources()
        merged = resources.merge(ncpu_calib=8, ncpu_trap=2)
        
        assert resources.ncpu_calib == 4  # Original unchanged
        assert merged.ncpu_calib == 8  # Merged changed
        assert merged.ncpu_trap == 2
        assert merged.ncpu_extract == 4  # Unchanged field


class TestDirectoryConfig:
    """Test the DirectoryConfig class."""

    def test_directory_config_defaults(self):
        """Test that DirectoryConfig has correct default paths."""
        config = DirectoryConfig()
        
        expected_base = Path.home() / "data/sphere"
        assert config.base_path == expected_base
        assert config.raw_directory == expected_base / "data"
        assert config.reduction_directory == expected_base / "reduction"

    def test_directory_config_custom_base(self):
        """Test DirectoryConfig with custom base path."""
        custom_base = Path("/tmp/test_sphere")
        config = DirectoryConfig(base_path=custom_base)
        
        assert config.base_path == custom_base
        assert config.raw_directory == custom_base / "data"
        assert config.reduction_directory == custom_base / "reduction"

    def test_directory_config_string_paths(self):
        """Test DirectoryConfig with string paths."""
        config = DirectoryConfig(
            base_path="/tmp/sphere",
            raw_directory="/tmp/sphere/raw_data",
            reduction_directory="/tmp/sphere/reductions"
        )
        
        assert config.base_path == Path("/tmp/sphere")
        assert config.raw_directory == Path("/tmp/sphere/raw_data")
        assert config.reduction_directory == Path("/tmp/sphere/reductions")

    def test_directory_config_get_paths_dict(self):
        """Test that get_paths_dict() returns correct dictionary."""
        config = DirectoryConfig()
        paths_dict = config.get_paths_dict()
        
        assert isinstance(paths_dict, dict)
        assert "base_path" in paths_dict
        assert "raw_directory" in paths_dict
        assert "reduction_directory" in paths_dict
        assert all(isinstance(path, Path) for path in paths_dict.values())

    def test_directory_config_merge(self):
        """Test that DirectoryConfig.merge() works correctly."""
        config = DirectoryConfig()
        custom_raw = Path("/custom/raw")
        merged = config.merge(raw_directory=custom_raw)
        
        assert config.raw_directory != custom_raw  # Original unchanged
        assert merged.raw_directory == custom_raw  # Merged changed


class TestPipelineStepsConfig:
    """Test the PipelineStepsConfig class."""

    def test_pipeline_steps_defaults(self):
        """Test that PipelineStepsConfig has correct default values."""
        config = PipelineStepsConfig()
        
        # Most steps should be enabled by default
        assert config.download_data is True
        assert config.reduce_calibration is True
        assert config.extract_cubes is True
        assert config.run_trap_reduction is True
        
        # Some optional steps disabled by default
        assert config.bundle_hexagons is False
        assert config.bundle_residuals is False

    def test_pipeline_steps_enable_all_ifs(self):
        """Test enable_all_ifs_steps() method."""
        config = PipelineStepsConfig()
        config.disable_all_ifs_steps()  # First disable all
        
        # Verify some are disabled
        assert config.download_data is False
        assert config.reduce_calibration is False
        
        config.enable_all_ifs_steps()
        
        # Verify IFS steps are enabled
        assert config.download_data is True
        assert config.reduce_calibration is True
        assert config.extract_cubes is True
        
        # TRAP steps should not be affected
        assert config.run_trap_reduction is True  # Not in IFS steps list

    def test_pipeline_steps_disable_all_ifs(self):
        """Test disable_all_ifs_steps() method."""
        config = PipelineStepsConfig()
        config.disable_all_ifs_steps()
        
        # Verify IFS steps are disabled
        assert config.download_data is False
        assert config.reduce_calibration is False
        assert config.extract_cubes is False
        
        # TRAP steps should not be affected
        assert config.run_trap_reduction is True  # Not in IFS steps list

    def test_pipeline_steps_all_steps_disabled(self):
        """Test all_steps_disabled() method."""
        config = PipelineStepsConfig()
        assert config.all_steps_disabled() is False  # Some enabled by default
        
        config.disable_all_ifs_steps()
        assert config.all_steps_disabled() is True

    def test_pipeline_steps_merge(self):
        """Test that PipelineStepsConfig.merge() works correctly."""
        config = PipelineStepsConfig()
        merged = config.merge(
            download_data=False,
            run_trap_detection=False,
            overwrite_calibration=False
        )
        
        assert config.download_data is True  # Original unchanged
        assert merged.download_data is False  # Merged changed
        assert merged.run_trap_detection is False
        assert merged.overwrite_calibration is False


class TestIFSReductionConfig:
    """Test the main IFSReductionConfig class."""

    def test_ifs_reduction_config_instantiation(self):
        """Test that IFSReductionConfig can be instantiated with defaults."""
        config = IFSReductionConfig()
        
        # Check that all sub-configs are present
        assert isinstance(config.calibration, CalibrationConfig)
        assert isinstance(config.extraction, ExtractionConfig)
        assert isinstance(config.preprocessing, PreprocConfig)
        assert isinstance(config.directories, DirectoryConfig)
        assert isinstance(config.resources, Resources)
        assert isinstance(config.steps, PipelineStepsConfig)

    def test_ifs_reduction_config_as_plain_dicts(self):
        """Test that as_plain_dicts() returns tuple of dictionaries."""
        config = IFSReductionConfig()
        dicts = config.as_plain_dicts()
        
        assert isinstance(dicts, tuple)
        assert len(dicts) == 4  # calibration, extraction, preprocessing, directories
        assert all(isinstance(d, dict) for d in dicts)

    def test_ifs_reduction_config_apply_resources(self):
        """Test that apply_resources() correctly applies resource settings."""
        config = IFSReductionConfig()
        config.resources.ncpu = 8
        config.apply_resources()
        
        assert config.calibration.ncpus == 8
        assert config.preprocessing.ncpu_cubebuilding == 8
        assert config.preprocessing.ncpu_find_center == 8

    def test_ifs_reduction_config_set_ncpu(self):
        """Test that set_ncpu() convenience method works."""
        config = IFSReductionConfig()
        config.set_ncpu(12)
        
        # Check that resources are set
        assert config.resources.ncpu == 12
        
        # Check that they're applied to sub-configs
        assert config.calibration.ncpus == 12
        assert config.preprocessing.ncpu_cubebuilding == 12

    def test_ifs_reduction_config_apply_trap_resources(self):
        """Test apply_trap_resources method with mock trap config."""
        config = IFSReductionConfig()
        config.resources.ncpu_trap = 6
        
        # Create a mock trap config object
        class MockTrapConfig:
            def __init__(self):
                self.resources = MockResources()
            
            def apply_resources(self):
                pass
        
        class MockResources:
            def __init__(self):
                self.ncpu_reduction = None
        
        trap_config = MockTrapConfig()
        config.apply_trap_resources(trap_config)
        
        assert trap_config.resources.ncpu_reduction == 6


def test_default_ifs_reduction():
    """Test that defaultIFSReduction() factory function works."""
    config = defaultIFSReduction()
    assert isinstance(config, IFSReductionConfig)
    
    # Should have same structure as direct instantiation
    direct_config = IFSReductionConfig()
    assert isinstance(config.calibration, CalibrationConfig)
    assert isinstance(config.extraction, ExtractionConfig)
    assert isinstance(direct_config.calibration, CalibrationConfig)
    assert isinstance(direct_config.extraction, ExtractionConfig)


@pytest.mark.parametrize("config_class", [
    CalibrationConfig,
    ExtractionConfig,
    PreprocConfig,
    DirectoryConfig,
    Resources,
    PipelineStepsConfig,
])
def test_config_classes_have_merge_method(config_class):
    """Test that all config classes have a merge method."""
    config = config_class()
    assert hasattr(config, 'merge')
    assert callable(config.merge)
    
    # Test that merge returns the same type
    merged = config.merge()
    assert isinstance(merged, config_class)

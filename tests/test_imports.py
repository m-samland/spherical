"""
Test basic imports for the spherical package.

This module ensures that all major components of the spherical package
can be imported without errors, which is essential for verifying that
the package is properly installed and all dependencies are available.
"""

import pytest


def _check_optional_dependency(module_name):
    """Check if an optional dependency is available."""
    try:
        import importlib
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def test_import_spherical():
    """Test that the main spherical package can be imported."""
    import spherical
    assert hasattr(spherical, '__version__')


def test_import_spherical_database():
    """Test that spherical.database submodule can be imported."""
    import spherical.database  # noqa: F401
    import spherical.database.sphere_database  # noqa: F401
    import spherical.database.file_table  # noqa: F401
    import spherical.database.observation_table  # noqa: F401
    import spherical.database.target_table  # noqa: F401
    import spherical.database.ifs_observation  # noqa: F401
    import spherical.database.irdis_observation  # noqa: F401


def test_import_spherical_database_components():
    """Test that specific database components can be imported."""
    from spherical.database.sphere_database import SphereDatabase
    from spherical.database.file_table import make_file_table
    from spherical.database.observation_table import create_observation_table
    from spherical.database.target_table import make_target_list_with_SIMBAD
    
    # Verify classes/functions are importable
    assert SphereDatabase is not None
    assert make_file_table is not None
    assert create_observation_table is not None
    assert make_target_list_with_SIMBAD is not None


def test_import_spherical_pipeline():
    """Test that spherical.pipeline submodule can be imported."""
    import spherical.pipeline  # noqa: F401
    import spherical.pipeline.pipeline_config  # noqa: F401
    import spherical.pipeline.logging_utils  # noqa: F401
    import spherical.pipeline.toolbox  # noqa: F401
    import spherical.pipeline.transmission  # noqa: F401


def test_import_spherical_pipeline_components():
    """Test that specific pipeline components can be imported."""
    from spherical.pipeline.pipeline_config import (
        IFSReductionConfig,
        CalibrationConfig,
        ExtractionConfig,
        PreprocConfig,
        DirectoryConfig,
        Resources,
        PipelineStepsConfig,
        defaultIFSReduction
    )
    
    # Verify classes/functions are importable
    assert IFSReductionConfig is not None
    assert CalibrationConfig is not None
    assert ExtractionConfig is not None
    assert PreprocConfig is not None
    assert DirectoryConfig is not None
    assert Resources is not None
    assert PipelineStepsConfig is not None
    assert defaultIFSReduction is not None


@pytest.mark.skipif(
    not _check_optional_dependency("charis"),
    reason="charis dependency not available"
)
def test_import_spherical_pipeline_ifs_reduction():
    """Test that IFS reduction module can be imported (requires optional dependencies)."""
    from spherical.pipeline.ifs_reduction import execute_targets
    assert execute_targets is not None


@pytest.mark.skipif(
    not _check_optional_dependency("charis"), 
    reason="charis dependency not available"
)
def test_import_spherical_pipeline_steps():
    """Test that pipeline steps submodule can be imported (requires optional dependencies)."""
    import spherical.pipeline.steps  # noqa: F401
    import spherical.pipeline.steps.find_star  # noqa: F401
    import spherical.pipeline.steps.extract_cubes  # noqa: F401


def test_import_spherical_utils():
    """Test that spherical.utils submodule can be imported."""
    import spherical.utils  # noqa: F401


def test_import_spherical_scripts():
    """Test that spherical.scripts submodule can be imported."""
    import spherical.scripts  # noqa: F401


@pytest.mark.parametrize("module_name", [
    "spherical",
    "spherical.database",
    "spherical.pipeline", 
    "spherical.utils",
    "spherical.scripts"
])
def test_import_all_main_modules(module_name):
    """Test that all main spherical modules can be imported."""
    import importlib
    module = importlib.import_module(module_name)
    assert module is not None


def test_import_version():
    """Test that version information is accessible."""
    import spherical
    version = spherical.__version__
    assert isinstance(version, str)
    assert len(version) > 0
    # Should either be a proper version string or "dev"
    assert version == "dev" or "." in version

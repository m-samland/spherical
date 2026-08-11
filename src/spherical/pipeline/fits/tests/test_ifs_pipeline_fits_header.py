import logging
import os
from pathlib import Path
from typing import Literal

import pytest
from astropy.io.fits.header import Header

from spherical.pipeline.fits import headers

LOGGER = logging.getLogger(__name__)
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def _get_spherical_test_data_directory() -> Path:
    """
    Locate the package's 'tests/data' folder or skip dependent tests if not present.

    Returns
    -------
    Path
        Root path to 'spherical/tests/data'.

    Raises
    ------
    pytest.skip
        If the directory does not exist or is not a folder.
    """
    import spherical
    spherical_repo_root = Path(spherical.__file__).resolve().parents[2]
    test_data_dir = spherical_repo_root / "tests/data"

    try:
        assert test_data_dir.exists(), "Test data directory does not exist. Cannot run tests that require test data."
        assert test_data_dir.is_dir(), "Test data path is not a directory. Cannot run tests that require test data."
    except AssertionError as e:
        pytest.skip(f"Skipping test because {e}.")

    return test_data_dir


class TestUnitExtendFitsHeaderWithCard:
    """
    Unit tests for the 'extend_fits_header_with_card' function.
    """

    @pytest.mark.filterwarnings("ignore::astropy.io.fits.verify.VerifyWarning")
    @pytest.mark.usefixtures("tmp_path_factory")
    @pytest.mark.parametrize(
        "key",
        [
            "TEST",
            "TEST KEY"
            "TEST_KEY",
            "__TEST__",
        ],
    )
    @pytest.mark.parametrize(
        "value",
        [
            0,
            0.0,
            1.0e5,
            "test_value",
            "test value",
            "test value with spaces",
            "test_value_with_underscores",
            "test-value-with-dashes",
            "test:value:with:colons",
            "test/value/with/slashes",
            "test\\value\\with\\backslashes",
            "test.value.with.dots",
            "test,value,with,commas",
            "test@value@with@at-signs",
            "test_special!@#$%^&*()_+-=[]{}|;':\",.<>?/~`",
        ],
    )
    @pytest.mark.parametrize(
        "comment",
        [
            "This is a test comment.",
            "Another test comment.",
            "Comment with special characters: !@#$%^&*()",
            "Comment with spaces and punctuation.",
        ],
    )
    def test_extend_fits_header_with_card_basic(self, tmp_path_factory, key: str, value: str | int | float, comment: str):
        """
        Test the basic functionality of 'extend_fits_header_with_card'.
        """

        tmp_dir = tmp_path_factory.mktemp(f"{Path(__file__).stem}")
        tmp_file_path = tmp_dir / "test_extend_fits_header_with_card_basic.fits"

        hdr = Header()

        headers.extend_fits_header_with_card(
            hdr, 
            key=key, 
            value=value, 
            comment=comment, 
            update=True
        )

        assert hdr[f'HIERARCH {key}'] == value
        assert hdr.comments[f'HIERARCH {key}'] == comment

        with open(tmp_file_path, 'wb') as f:
            hdr.tofile(f)


class TestIntegrationUpdateFitsHeaderAfterReduction:
    """
    Integration tests for updating FITS headers after IFS reduction.
    """

    # TODO: this should copy the test data to a temporary directory and work on that copy in a function scope fixture
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not run test in GitHub CI.")
    @pytest.mark.filterwarnings("ignore::astropy.io.fits.verify.VerifyWarning")
    @pytest.mark.parametrize(
        ("target_observation_name"),
        [
            # TODO: segregate between GitHub CI and local tests
            # pytest.param(
            #     "PATH_TO_TEST_DATA", 
            #     marks=pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Do not run test in GitHub.")
            # ),
            "PATH_TO_TEST_DATA", 
        ],
    )
    @pytest.mark.parametrize(
        ("target"),
        ["coro", "center", "flux", "all"],
    )
    @pytest.mark.parametrize(
        ("override_mode_file"),
        [
            "update",
            pytest.param(
                "copy", 
                marks=pytest.mark.xfail(reason="'copy' mode not implemented yet")
            ),
        ],
    )
    def test_make_fits_header_from_ifs_reduction(
        self,
        caplog, 
        target_observation_name: str, 
        target: Literal["coro", "center", "flux", "all"], 
        override_mode_file: Literal["copy", "update"],
        override_mode_header: Literal["keep", "update"] = "update",
        multiple_match: Literal["run_all", "skip", "fail"] = "fail",
    ):
        """
        Test 'update_cube_fits_header_after_reduction' on real pipeline outputs.

        Parametrize over observation names, targets, and override modes,
        handling GitHub CI skips, expected failures, and multiple-directory logic.

        Parameters
        ----------
        caplog
            Pytest fixture for capturing logs.
        target_observation_name : str
            Name of the test reduction directory to locate.
        target : {"coro","center","flux","all"}
            Which FITS cube variant to update.
        override_mode_file : {"copy","update"}
            Whether to copy or update the file (copy is xfail).
        override_mode_header : {"keep","update"}
            Whether to preserve or overwrite existing headers.
        multiple_match : {"run_all","skip","fail"}, default "fail"
            How to handle finding multiple matching test dirs.
        """

        if target_observation_name == "PATH_TO_TEST_DATA":
            pytest.fail(
                reason="TODO: Please set 'target_observation_name' to a valid test observation name or path to test data. RE: issue https://github.com/m-samland/spherical/issues/61"
            )

        # get root directory of test observation(s)
        spherical_test_data_dir = _get_spherical_test_data_directory()
        reduction_dir = list(spherical_test_data_dir.rglob(target_observation_name))

        if len(reduction_dir) == 0:
            pytest.skip(f"Skipping test because no reduction directory found for {target_observation_name}.")
        elif len(reduction_dir) > 1 and (multiple_match == "fail"):
            pytest.fail(f"Expected exactly one reduction directory for {target_observation_name}, found {len(reduction_dir)}. Failing test because multiple directories found and 'multiple_match' is set to 'fail'. Set 'multiple_match' to 'run_all' or 'skip' to allow multiple directories.")
        elif len(reduction_dir) > 1 and (multiple_match == "skip"):
            pytest.skip(f"Skipping test because multiple reduction directories found for {target_observation_name}. Set 'multiple_match' to 'run_all' or 'fail' to run the test with multiple directories.")
        elif len(reduction_dir) > 1 and (multiple_match == "run_all"):
            LOGGER.warning(f"Multiple reduction directories found for {target_observation_name}. Running test with all directories. Set 'multiple_match' to 'fail' or 'skip' to change this behavior.")
            for dir in reduction_dir:
                LOGGER.debug(f"Running test with directory: {dir}")
                self.test_make_fits_header_from_ifs_reduction(caplog, dir.stem, multiple_match=multiple_match)
            return
        elif len(reduction_dir) == 1:
            reduction_dir = reduction_dir[0]
        else:
            pytest.fail(f"Unexpected number of reduction directories found for {target_observation_name}. Expected 1, found {len(reduction_dir)}.")

        LOGGER.debug(f"Resolved test reduction directory: {reduction_dir}")

        # run the function to test
        headers.update_cube_fits_header_after_reduction(
            path=reduction_dir,
            target=target,
            override_mode_file=override_mode_file,
            override_mode_header=override_mode_header,
        )

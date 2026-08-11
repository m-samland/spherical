"""Tests for the resume functionality in file table generation.

These tests simulate interrupted downloads by running make_file_table for
a short time window, then extending the date range on a second call. The
resume logic should detect DP.IDs in the existing output and only download
new ones.

The ESO traffic is replayed from cassettes (see cassettes/README.md), so these
run offline in the default suite. They took ~15 min live, which meant they were
deselected, which meant the resume logic was covered by nothing anyone ran.
`test_live_archive_still_responds` stays live as the canary for API drift.

The 2020-03-07 window is chosen, not incidental: header retrieval is one HTTP GET
per DP.ID, so the window size *is* the cost. That night holds 6 files of a single
target (A124420), and extending to 2020-03-09 adds 2 WAVE,LAMP calibrations — 6
and 8 requests, against 171 and 305 for the 2016-09-15 window this replaced.
Six files also halve cleanly for the simulated-partial test.

Note ESO's stime/etime are night-based, not calendar: [2020-03-07, 2020-03-08]
selects the night beginning 2020-03-07, whose exposures carry 2020-03-08 DATE_OBS.
"""

import pandas as pd
import pytest

from spherical.database.file_table import make_file_table

INSTRUMENT = "ifs"


@pytest.fixture(scope="module")
def resume_table_path(tmp_path_factory):
    return tmp_path_factory.mktemp("resume_test")


@pytest.mark.remote_data
def test_live_archive_still_responds(tmp_path):
    """Canary: the cassettes are only trustworthy while the real API still matches them.

    Deliberately asserts little — its job is to fail when ESO changes the query
    interface or the 2020-03-07 holdings, which is exactly when the replayed tests
    below stop meaning anything.
    """
    table = make_file_table(
        output_dir=tmp_path,
        instrument=INSTRUMENT,
        start_date="2020-03-07",
        end_date="2020-03-08",
        output_suffix="_canary",
        cache=False,
        batch_size=100,
        date_batch_months=1,
    )

    assert len(table) > 0, "Live ESO archive returned no SPHERE IFS files for the night of 2020-03-07"
    assert "DP.ID" in table.colnames


@pytest.mark.vcr
class TestResumeFiletable:
    """Test suite for the resume/incremental file table building."""

    def test_fresh_run_creates_output_no_partial(self, resume_table_path):
        """A complete fresh run should produce an output file and no partial file."""
        suffix = "_fresh"
        table = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-08",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
        )

        output_path = resume_table_path / f"table_of_files_{INSTRUMENT}{suffix}.csv"
        partial_path = resume_table_path / f"table_of_files_{INSTRUMENT}{suffix}_partial.csv"

        assert output_path.exists(), "Output file should exist after fresh run"
        assert not partial_path.exists(), "Partial file should be cleaned up after successful run"
        assert len(table) > 0, "Table should have entries for this date range"

    def test_incremental_extends_date_range(self, resume_table_path):
        """Running again with a wider date range should add new entries via resume."""
        suffix = "_incr"

        # First run: one day
        table1 = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-08",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
        )
        n_first = len(table1)
        assert n_first > 0

        # Second run: extended by one day (resume=True by default)
        table2 = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-09",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
        )

        assert len(table2) >= n_first, "Extended run should have at least as many entries"
        # Check no duplicates
        dp_ids = list(table2["DP.ID"])
        assert len(dp_ids) == len(set(dp_ids)), "No duplicate DP.IDs in merged table"

    def test_resume_with_simulated_partial_file(self, resume_table_path):
        """Simulates an interrupted run by manually creating a partial file,
        then verifies that resume picks it up and merges correctly."""
        suffix = "_simpartial"

        # First run to get real data
        table1 = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-08",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
        )
        n_first = len(table1)
        assert n_first > 0

        output_path = resume_table_path / f"table_of_files_{INSTRUMENT}{suffix}.csv"
        partial_path = resume_table_path / f"table_of_files_{INSTRUMENT}{suffix}_partial.csv"

        # Read the output and split it: put half in partial, keep half in output
        # This simulates: some data was downloaded in a previous complete run,
        # and additional data was partially downloaded before interruption
        df = pd.read_csv(output_path)
        mid = len(df) // 2
        df_output = df.iloc[:mid]
        df_partial = df.iloc[mid:]

        df_output.to_csv(output_path, index=False)
        df_partial.to_csv(partial_path, index=False)

        # Now run with resume=True — should merge both and extend
        table2 = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-09",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
            resume=True,
        )

        # Partial file should be cleaned up
        assert not partial_path.exists(), "Partial file should be removed after merge"
        # Should have at least all original entries
        assert len(table2) >= n_first
        # No duplicates
        dp_ids = list(table2["DP.ID"])
        assert len(dp_ids) == len(set(dp_ids)), "No duplicate DP.IDs after resume merge"

    def test_resume_false_ignores_partial(self, resume_table_path):
        """When resume=False, any existing partial file should be deleted."""
        suffix = "_noresume"

        # Create a dummy partial file
        partial_path = resume_table_path / f"table_of_files_{INSTRUMENT}{suffix}_partial.csv"
        pd.DataFrame({"DP.ID": ["FAKE.001"], "MJD_OBS": [57000.0]}).to_csv(partial_path, index=False)

        table = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-08",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
            resume=False,
        )

        assert not partial_path.exists(), "Partial file should be deleted when resume=False"
        # The fake entry should NOT be in the final table
        dp_ids = list(table["DP.ID"])
        assert "FAKE.001" not in dp_ids, "Fake partial data should not appear when resume=False"

    def test_idempotent_rerun(self, resume_table_path):
        """Running the same date range twice should produce the same result."""
        suffix = "_idem"

        table1 = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-08",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
        )

        table2 = make_file_table(
            output_dir=resume_table_path,
            instrument=INSTRUMENT,
            start_date="2020-03-07",
            end_date="2020-03-08",
            output_suffix=suffix,
            cache=False,
            batch_size=100,
            date_batch_months=1,
        )

        # Without this the test is vacuous: query_eso_data swallows request failures and
        # returns nothing, so two empty tables would compare equal and pass. That is not
        # hypothetical — it is exactly what a partially-recorded cassette produced here.
        assert len(table1) > 0, "First run returned no files; the comparison below would be vacuous"

        assert len(table1) == len(table2), "Rerun with same range should produce same number of entries"
        assert set(table1["DP.ID"]) == set(table2["DP.ID"]), "Same DP.IDs on rerun"

"""Tests for the multi-epoch target selector (design spec 2026-07-22, section 1.1)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from spherical.database.multi_epoch_filter import (
    BG_MOTION_COLUMN,
    N_EPOCHS_COLUMN,
    SPAN_COLUMN,
    read_host_list,
    select_multi_epoch_targets,
)


def make_table(rows, *, night_start_bytes=False):
    """Build an observation table from ``(main_id, night_start, pmra, pmdec)`` rows."""
    main_id, night_start, pmra, pmdec = zip(*rows)
    table = Table()
    table["MAIN_ID"] = list(main_id)
    table["NIGHT_START"] = (
        np.array([n.encode() for n in night_start], dtype="S10") if night_start_bytes else list(night_start)
    )
    table["PMRA"] = np.array(pmra, dtype=float)
    table["PMDEC"] = np.array(pmdec, dtype=float)
    return table


# pm_total = hypot(3, 4) = 5 mas/yr; over 3 yr that is 15 mas = 1.22 px at 12.25 mas/px.
THREE_YEAR_PAIR = [
    ("HD 3795", "2020-01-01", 3.0, 4.0),
    ("HD 3795", "2023-01-01", 3.0, 4.0),
]


class TestSelectMultiEpochTargets:
    def test_pair_spanning_three_years_passes(self):
        selected = select_multi_epoch_targets(make_table(THREE_YEAR_PAIR))
        assert len(selected) == 2
        assert selected[BG_MOTION_COLUMN][0] == pytest.approx(1.2249, abs=1e-3)

    def test_short_span_drops(self):
        rows = [
            ("HD 3795", "2020-01-01", 3.0, 4.0),
            ("HD 3795", "2020-02-06", 3.0, 4.0),  # 0.1 yr -> 0.5 mas
        ]
        assert len(select_multi_epoch_targets(make_table(rows))) == 0

    def test_same_night_repeat_drops(self):
        rows = [
            ("HD 3795", "2020-01-01", 300.0, 400.0),
            ("HD 3795", "2020-01-01", 300.0, 400.0),
        ]
        assert len(select_multi_epoch_targets(make_table(rows))) == 0

    def test_single_epoch_drops_despite_huge_proper_motion(self):
        rows = [("Barnard's star", "2020-01-01", 8000.0, 4000.0)]
        assert len(select_multi_epoch_targets(make_table(rows))) == 0

    def test_masked_proper_motion_drops_group_with_warning(self, caplog):
        table = make_table(THREE_YEAR_PAIR)
        table["PMRA"] = MaskedColumn(table["PMRA"], mask=[False, True])
        with caplog.at_level(logging.WARNING):
            selected = select_multi_epoch_targets(table)
        assert len(selected) == 0
        assert "HD 3795" in caplog.text

    def test_nan_proper_motion_drops_group(self):
        table = make_table(THREE_YEAR_PAIR)
        table["PMDEC"][1] = np.nan
        assert len(select_multi_epoch_targets(table)) == 0

    def test_stricter_threshold_drops_the_marginal_case(self):
        selected = select_multi_epoch_targets(make_table(THREE_YEAR_PAIR), min_bg_motion_px=3.0)
        assert len(selected) == 0

    def test_byte_string_night_start_parses(self):
        """The real observation table stores NIGHT_START as |S10."""
        table = make_table(THREE_YEAR_PAIR, night_start_bytes=True)
        assert table["NIGHT_START"].dtype.kind == "S"
        assert len(select_multi_epoch_targets(table)) == 2

    def test_empty_input_gives_empty_output(self):
        table = Table(
            names=("MAIN_ID", "NIGHT_START", "PMRA", "PMDEC"),
            dtype=("U16", "U10", float, float),
        )
        selected = select_multi_epoch_targets(table)
        assert len(selected) == 0
        assert SPAN_COLUMN in selected.colnames

    def test_metadata_columns_are_constant_per_target(self):
        rows = THREE_YEAR_PAIR + [("HD 3795", "2024-01-01", 3.0, 4.0)]
        selected = select_multi_epoch_targets(make_table(rows))
        assert len(selected) == 3
        assert list(selected[N_EPOCHS_COLUMN]) == [3, 3, 3]
        assert selected[SPAN_COLUMN][0] == pytest.approx(4.0, abs=1e-2)
        assert len(set(selected[BG_MOTION_COLUMN])) == 1

    def test_rows_keep_input_order_across_interleaved_targets(self):
        rows = [
            ("HD 3795", "2020-01-01", 3.0, 4.0),
            ("Solo", "2020-06-01", 3000.0, 0.0),
            ("HD 3795", "2023-01-01", 3.0, 4.0),
        ]
        selected = select_multi_epoch_targets(make_table(rows))
        assert list(selected["NIGHT_START"]) == ["2020-01-01", "2023-01-01"]

    def test_missing_column_raises(self):
        table = make_table(THREE_YEAR_PAIR)
        del table["PMDEC"]
        with pytest.raises(KeyError, match="PMDEC"):
            select_multi_epoch_targets(table)

    def test_several_rows_on_one_night_count_as_one_epoch(self):
        """Two modes on one night plus a later night is two epochs, not three rows."""
        table = make_table(
            [
                ("HD 3795", "2020-01-01", 3.0, 4.0),
                ("HD 3795", "2020-01-01", 3.0, 4.0),
                ("HD 3795", "2023-01-01", 3.0, 4.0),
            ]
        )
        assert len(select_multi_epoch_targets(table, min_epochs=3)) == 0
        selected = select_multi_epoch_targets(table, min_epochs=2)
        assert len(selected) == 3
        assert list(selected[N_EPOCHS_COLUMN]) == [2, 2, 2]

    def test_min_epochs_can_be_raised(self):
        rows = THREE_YEAR_PAIR + [("HD 3795", "2024-01-01", 3.0, 4.0)]
        assert len(select_multi_epoch_targets(make_table(rows), min_epochs=4)) == 0
        assert len(select_multi_epoch_targets(make_table(rows), min_epochs=3)) == 3


class TestReadHostList:
    def test_skips_blanks_and_comments(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text(
            "# known exoplanet hosts\n"
            "beta Pic\n"
            "\n"
            "   \n"
            "HR 8799   # four planets\n"
            "51 Eri\n"
        )
        assert read_host_list(path) == ["beta Pic", "HR 8799", "51 Eri"]

    def test_empty_file_gives_empty_list(self, tmp_path):
        path = tmp_path / "hosts.txt"
        path.write_text("")
        assert read_host_list(path) == []

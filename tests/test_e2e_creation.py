import numpy as np


def test_extended_table_covers_full_range(persistent_file_table):
    dates = [str(date) for date in persistent_file_table["DATE_SHORT"]]
    assert "2016-09-15" in dates
    assert "2016-09-16" in dates
    assert "2016-09-17" in dates

def test_database_creation(persistent_file_table, persistent_target_table, persistent_observation_table):
    assert len(persistent_file_table) > 0
    assert len(persistent_target_table) > 0
    assert len(persistent_observation_table) > 0
    assert np.all(persistent_observation_table['TOTAL_EXPTIME'] > 0)
    assert np.all(persistent_observation_table['ROTATION'] >= 0)
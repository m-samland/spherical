import numpy as np


def test_database_creation(persistent_master_table, persistent_target_table, persistent_observation_table):
    assert len(persistent_master_table) > 0
    assert len(persistent_target_table) > 0
    assert len(persistent_observation_table) > 0
    assert np.all(persistent_observation_table['TOTAL_EXPTIME'] > 0)
    assert np.all(persistent_observation_table['ROTATION'] >= 0)
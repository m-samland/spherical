def test_retrieve_observation_object_list(sphere_db, persistent_observation_SIMBAD_table):
    observation_table = persistent_observation_SIMBAD_table

    assert len(observation_table) > 0, "No observations of beta Pic found, even though one should exist."

    observation_objects = sphere_db.retrieve_observation_metadata(observation_table)

    assert isinstance(observation_objects, list)
    assert len(observation_objects) == len(observation_table)

    for obs_obj in observation_objects:
        assert obs_obj is not None
        assert len(obs_obj.all_frames) > 0
        assert len(obs_obj.frames['WAVECAL']) > 0
        assert len(obs_obj.frames['CENTER']) > 0
        assert len(obs_obj.frames['FLUX']) > 0

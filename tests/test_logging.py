def test_get_pipeline_logger(tmp_path):
    from spherical.pipeline.logging_utils import get_pipeline_logger, remove_queue_listener
    logger = get_pipeline_logger("demo", tmp_path)
    logger.info("hello")
    # ensure listener thread exists
    assert logger.handlers
    remove_queue_listener()

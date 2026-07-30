def test_get_pipeline_logger(tmp_path):
    from spherical.pipeline.logging_utils import get_pipeline_logger, remove_queue_listener
    logger = get_pipeline_logger("demo", tmp_path)
    logger.info("hello")
    # ensure listener thread exists
    assert logger.handlers
    remove_queue_listener()


def test_reused_logger_name_keeps_writing_a_jsonlog(tmp_path):
    """Two observations sharing a target/band/night reuse one logger name.

    The second must still leave a `reduction.jsonlog` in the folder, otherwise
    `aggregate_reduction_status` (which rglobs for that exact name) reports the
    target as missing entirely.
    """
    import time

    from spherical.pipeline.logging_utils import get_pipeline_logger, remove_queue_listener

    context = {
        "target": "51_Eri", "band": "DB_K12", "night": "2015-09-24",
        "step": "session_start", "status": "started",
    }

    for _ in range(2):
        logger = get_pipeline_logger("51_Eri/DB_K12/2015-09-24", tmp_path, verbose=False)
        logger.info("session started", extra=context)
        time.sleep(0.5)  # let the QueueListener thread drain before it is stopped
        remove_queue_listener()

    jsonlog = tmp_path / "reduction.jsonlog"
    assert jsonlog.exists()
    assert jsonlog.read_text().strip(), "records from the second session were dropped"

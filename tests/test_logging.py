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


def test_library_logger_records_reach_the_target_log(tmp_path):
    """trap logs its progress on its own logger tree, which nothing here
    configures — without the bridge those records die at `logging.lastResort`
    (level WARNING), so a target that runs for hours leaves a log holding only
    the pipeline's own bookends.
    """
    import logging
    import time

    from spherical.pipeline.logging_utils import (
        bridge_library_logger,
        get_pipeline_logger,
        remove_queue_listener,
    )

    library_logger = logging.getLogger("trap")
    level_before = library_logger.level
    handlers_before = list(library_logger.handlers)

    logger = get_pipeline_logger("HD_1/DB_H23/2024-06-11", tmp_path, verbose=False)
    with bridge_library_logger(logger, "trap"):
        logging.getLogger("trap.detection").info("running template matching")
    time.sleep(0.5)  # let the QueueListener thread drain
    remove_queue_listener()

    assert "running template matching" in (tmp_path / "reduction.log").read_text()
    # The bridge is scoped to one target and must leave no residue.
    assert library_logger.level == level_before
    assert library_logger.handlers == handlers_before

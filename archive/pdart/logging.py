import logging
import os.path
import pdslogger  # type: ignore
import sys


def init_logging() -> None:
    """
    A central place for configuring logging.
    """
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s %(name)r]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.DEBUG,
    )


def set_up_pds_logger():  # type: ignore
    # Set up PDS_LOGGER
    _proposal_id = str(sys.argv[1]).zfill(5)
    _log_path = os.path.join(
        os.environ["TMP_WORKING_DIR"], f"logs/hst_{_proposal_id}_pipeline_log.log"
    )
    _log_dir = os.path.join(os.environ["TMP_WORKING_DIR"], "logs")
    _info_handler = pdslogger.file_handler(
        _log_path, level=logging.INFO, rotation="ymdhms"
    )

    _error_handler = pdslogger.error_handler(_log_dir, rotation="none")
    _warning_handler = pdslogger.warning_handler(_log_dir, rotation="none")

    _LOGGER = pdslogger.PdsLogger(f"hst_{_proposal_id}_pipeline")
    _LOGGER.add_handler(_info_handler)
    _LOGGER.add_handler(_error_handler)
    _LOGGER.add_handler(_warning_handler)

    return _LOGGER


PDS_LOGGER = set_up_pds_logger()  # type: ignore

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


# Set up PDS_LOGGER
_proposal_id = str(sys.argv[1]).zfill(5)
_log_path = os.path.join(
    os.environ["TMP_WORKING_DIR"], f"logs/hst_{_proposal_id}_pipeline_log.log"
)
_info_handler = pdslogger.file_handler(_log_path, level=logging.INFO, rotation="ymdhms")
PDS_LOGGER = pdslogger.PdsLogger(f"hst_{_proposal_id}_pipeline")
PDS_LOGGER.add_handler(_info_handler)

import sys

from pdart.logging import init_logging, PDS_LOGGER
from pdart.pipeline.directories import Directories, make_directories
from pdart.pipeline.state_machine2 import StateMachine2

# -1: print out every message
INFO_MESSAGE_LIMIT = -1


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    init_logging()
    dirs = make_directories()
    try:
        PDS_LOGGER.open(
            f"Pipeline for proposal id: {proposal_id}",
            limits={
                "info": INFO_MESSAGE_LIMIT,
            },
        )
        state_machine = StateMachine2(dirs, proposal_id)
        state_machine.run()
    except Exception as e:
        PDS_LOGGER.exception(e)
    finally:
        PDS_LOGGER.close()


if __name__ == "__main__":
    run()

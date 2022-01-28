import sys

from pdart.logging import init_logging, PDS_LOGGER
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.StateMachine import StateMachine

# -1: print out every message
INFO_MESSAGE_LIMIT = -1


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    init_logging()
    dirs = make_directories()
    PDS_LOGGER.open(
        f"Pipeline for proposal id: {proposal_id}",
        limits={
            "info": INFO_MESSAGE_LIMIT,
        },
    )
    state_machine = StateMachine(dirs, proposal_id)
    state_machine.run()
    PDS_LOGGER.close()


if __name__ == "__main__":
    run()

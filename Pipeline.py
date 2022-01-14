import sys

from pdart.Logging import init_logging, PDS_LOGGER
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.StateMachine import StateMachine


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    init_logging()
    dirs = make_directories()
    PDS_LOGGER.open(f"Pipeline for proposal id: {proposal_id}")
    state_machine = StateMachine(dirs, proposal_id)
    state_machine.run()
    PDS_LOGGER.close()


if __name__ == "__main__":
    run()

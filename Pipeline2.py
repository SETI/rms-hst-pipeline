import sys

from pdart.logging import init_logging
from pdart.pipeline.directories import Directories, make_directories
from pdart.pipeline.state_machine2 import StateMachine2


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    init_logging()
    dirs = make_directories()
    state_machine = StateMachine2(dirs, proposal_id)
    state_machine.run()


if __name__ == "__main__":
    run()

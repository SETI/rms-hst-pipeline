import sys

from pdart.Logging import init_logging
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.StateMachine2 import StateMachine2


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    init_logging()
    dirs = make_directories()
    state_machine = StateMachine2(dirs, proposal_id)
    state_machine.run()


if __name__ == "__main__":
    run()

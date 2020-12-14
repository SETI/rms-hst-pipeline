import logging
import sys
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.StateMachine2 import StateMachine2


def init_logging() -> None:
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s %(name)r]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.DEBUG,
    )


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    init_logging()
    dirs = make_directories()
    state_machine = StateMachine2(dirs, proposal_id)
    state_machine.run()


if __name__ == "__main__":
    run()

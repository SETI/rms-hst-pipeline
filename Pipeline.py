import sys
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.StateMachine import StateMachine


def run() -> None:
    assert len(sys.argv) == 2, sys.argv
    proposal_id = int(sys.argv[1])
    dirs = make_directories()
    state_machine = StateMachine(dirs, proposal_id)
    state_machine.run()


if __name__ == "__main__":
    run()

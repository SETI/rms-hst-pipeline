import sys
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.StateMachine import StateMachine


def run() -> None:
    assert len(sys.argv) <= 3, sys.argv
    proposal_id = int(sys.argv[1])
    use_selected_suffixes = sys.argv[2]
    selected_suffixes = True if use_selected_suffixes == "-c" else False
    dirs = make_directories()
    state_machine = StateMachine(dirs, proposal_id, selected_suffixes)
    state_machine.run()


if __name__ == "__main__":
    run()

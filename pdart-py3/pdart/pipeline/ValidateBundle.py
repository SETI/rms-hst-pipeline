from subprocess import CompletedProcess, run
from pdart.pipeline.Stage import Stage


class ValidateBundle(Stage):
    def _run(self) -> None:
        completed_process: CompletedProcess = run(
            ["./validate-pdart", self.dirs.deliverable_dir(self.proposal_id)]
        )
        completed_process.check_returncode()
        raise Exception("Succeeded but leaving a failure marker to prevent retries.")

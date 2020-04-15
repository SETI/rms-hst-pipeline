from subprocess import CompletedProcess, run
from pdart.pipeline.Stage import Stage


class ValidateBundle(Stage):
    def _run(self) -> None:
        completed_process: CompletedProcess = run(
            ["./validate-pdart", self.deliverable_dir()]
        )
        completed_process.check_returncode()
        raise Exception("Succeeded but leaving a failure marker to prevent retries.")

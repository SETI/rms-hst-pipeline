from subprocess import CompletedProcess, run
import fs.path
from pdart.pipeline.Stage import Stage


class ValidateBundle(Stage):
    def _run(self) -> None:
        completed_process: CompletedProcess = run(
            [
                "./validate-pdart",
                self.deliverable_bundle_dir(),
                self.deliverable_dir(),
                self.validation_report_dir(),
            ]
        )
        completed_process.check_returncode()

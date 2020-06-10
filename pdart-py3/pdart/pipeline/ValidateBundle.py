from subprocess import CompletedProcess, run
import fs.path
from pdart.pipeline.Stage import MarkedStage


class ValidateBundle(MarkedStage):
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

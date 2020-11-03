from subprocess import CompletedProcess, run
import fs.path
from pdart.pipeline.Stage import MarkedStage


class ValidateBundle(MarkedStage):
    """
    Run the PDS4 validation tool on the deliverable bundle.  If there
    are errors, they will be found in validation_report.txt.
    """

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

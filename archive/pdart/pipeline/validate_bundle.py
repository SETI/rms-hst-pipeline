from subprocess import CompletedProcess, run
import fs.path
import os.path
import json
from pdart.pipeline.changes_dict import (
    CHANGES_DICT_NAME,
)
from pdart.pipeline.stage import MarkedStage


class ValidateBundle(MarkedStage):
    """
    Run the PDS4 validation tool on the deliverable bundle.  If there
    are errors, they will be found in validation_report.txt.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        if not os.path.isdir(self.deliverable_dir()):
            raise ValueError(f"Need {self.deliverable_dir()} for ValidateBundle.")

        completed_process: CompletedProcess = run(
            [
                "./validate-pdart",
                self.deliverable_bundle_dir(),
                self.deliverable_dir(),
                self.validation_report_dir(),
            ]
        )
        completed_process.check_returncode()

        changes_dict_path = os.path.join(working_dir, CHANGES_DICT_NAME)
        os.remove(changes_dict_path)
        if os.path.isfile(changes_dict_path):
            raise ValueError(f"{changes_dict_path} doesn't exist.")

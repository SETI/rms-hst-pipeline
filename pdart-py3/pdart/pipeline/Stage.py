import abc
import os
import os.path
import traceback

from pdart.pipeline.Directories import Directories
from pdart.pipeline.MarkerFile import BasicMarkerFile

FAILURE_MARKER: str = "LAST$FAILURE.txt"


class Stage(metaclass=abc.ABCMeta):
    def __init__(self, dirs: Directories, proposal_id: int) -> None:
        self._bundle_segment = f"hst_{proposal_id:05}"
        self._dirs = dirs
        self._proposal_id = proposal_id
        self._marker_file = BasicMarkerFile(self.working_dir())

    def class_name(self) -> str:
        return type(self).__name__

    ##############################

    def __call__(self) -> None:
        if not os.path.exists(self.working_dir()):
            os.makedirs(self.working_dir())
        failure_marker_filepath = os.path.join(self.working_dir(), FAILURE_MARKER)
        if not os.path.isfile(failure_marker_filepath):
            try:
                self._marker_file.set_marker_info(self.class_name(), "running")
                self._run()
                self._marker_file.set_marker_info(self.class_name(), "success")
            except Exception as e:
                self._marker_file.set_marker_info(self.class_name(), "failure")
                with open(failure_marker_filepath, "w") as f:
                    header = (
                        f"EXCEPTION raised by {self._bundle_segment}, "
                        f"stage {self.class_name()}:"
                    )
                    print("****", header, str(e))
                    print(header, file=f)
                    print(traceback.format_exc(), file=f)

    @abc.abstractmethod
    def _run(self) -> None:
        pass

    ##############################

    def working_dir(self) -> str:
        return self._dirs.working_dir(self._proposal_id)

    def mast_downloads_dir(self) -> str:
        return self._dirs.mast_downloads_dir(self._proposal_id)

    def primary_files_dir(self) -> str:
        return self._dirs.primary_files_dir(self._proposal_id)

    def documents_dir(self) -> str:
        return self._dirs.documents_dir(self._proposal_id)

    def archive_primary_deltas_dir(self) -> str:
        return self._dirs.archive_primary_deltas_dir(self._proposal_id)

    def archive_browse_deltas_dir(self) -> str:
        return self._dirs.archive_browse_deltas_dir(self._proposal_id)

    def archive_label_deltas_dir(self) -> str:
        return self._dirs.archive_label_deltas_dir(self._proposal_id)

    def archive_dir(self) -> str:
        return self._dirs.archive_dir(self._proposal_id)

    def deliverable_dir(self) -> str:
        return self._dirs.deliverable_dir(self._proposal_id)

    def deliverable_bundle_dir(self) -> str:
        return self._dirs.deliverable_bundle_dir(self._proposal_id)

    def manifest_dir(self) -> str:
        return self._dirs.manifest_dir(self._proposal_id)

    def validation_report_dir(self) -> str:
        return self._dirs.validation_report_dir(self._proposal_id)

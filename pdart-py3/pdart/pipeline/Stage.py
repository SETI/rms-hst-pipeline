import abc
import os
import traceback

from pdart.pipeline.Directories import Directories

FAILURE_MARKER: str = "LAST$FAILURE.txt"


class Stage(metaclass=abc.ABCMeta):
    def __init__(self, dirs: Directories, proposal_id: int) -> None:
        self._bundle_segment = f"hst_{proposal_id:05}"
        self._dirs = dirs
        self._proposal_id = proposal_id

    def class_name(self) -> str:
        return type(self).__name__

    ##############################

    def __call__(self) -> None:
        failure_marker_filepath = os.path.join(self.working_dir(), FAILURE_MARKER)
        if not os.path.isfile(failure_marker_filepath):
            try:
                self._run()
            except Exception as e:
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

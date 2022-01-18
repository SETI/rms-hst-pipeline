import abc
import logging
import os
import os.path
import traceback
from typing import Optional

from pdart.pipeline.Directories import Directories
from pdart.pipeline.MarkerFile import BasicMarkerFile
from pdart.Logging import PDS_LOGGER


class Stage(metaclass=abc.ABCMeta):
    """
    Abstract class representing a stage of processing in the pipeline.
    The stage is intended to run as a transaction: that is, if it
    fails while running, it should roll back the state of the
    directory to what it looked like when the stage started.  This
    keeps garbage from one failed attempt from affecting future
    attempts.

    The class also accepts a list of directory paths that the stages
    will use and provides them to the operation of the stage.
    """

    def __init__(self, dirs: Directories, proposal_id: int) -> None:
        self._bundle_segment = f"hst_{proposal_id:05}"
        self._dirs = dirs
        self._proposal_id = proposal_id

    ##############################

    def __call__(self) -> None:
        self.begin_transaction()
        try:
            self._run()
            self.commit_transaction()
        except Exception as e:
            self.rollback_transaction(e)

    def begin_transaction(self) -> None:
        pass

    def commit_transaction(self) -> None:
        pass

    def rollback_transaction(self, e: Exception) -> None:
        pass

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

    def log_dir(self) -> str:
        return self._dirs.log_dir(self._proposal_id)


class MarkedStage(Stage):
    """
    A MarkedStage is a Stage that runs with a BasicMarkerFile to show
    its progress.
    """

    def __init__(self, dirs: Directories, proposal_id: int) -> None:
        Stage.__init__(self, dirs, proposal_id)
        if not os.path.exists(self.working_dir()):
            os.makedirs(self.working_dir())
        self._marker_file = BasicMarkerFile(self.working_dir())

    def class_name(self) -> str:
        return type(self).__name__

    def __call__(self) -> None:
        marker_info = self._marker_file.get_marker()
        if marker_info and marker_info.state == "FAILURE":
            return
        Stage.__call__(self)

    def begin_transaction(self) -> None:
        self._marker_file.set_marker_info(self.class_name(), "running")

    def commit_transaction(self) -> None:
        self._marker_file.set_marker_info(self.class_name(), "success")

    def rollback_transaction(self, e: Exception) -> None:
        error_text = (
            f"EXCEPTION raised by {self._bundle_segment}, "
            f"stage {self.class_name()}: {e}\n"
            f"{traceback.format_exc()}"
        )
        PDS_LOGGER.open(f"Stage '{self.class_name}' error raised")
        PDS_LOGGER.error(error_text)
        PDS_LOGGER.close()
        self._marker_file.set_marker_info(self.class_name(), "failure", error_text)

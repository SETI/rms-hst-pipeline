import abc
import logging
import os
from typing import Optional

from pdart.pipeline.stage import Stage
from pdart.pipeline.directories import Directories
from pdart.pipeline.marker_file import BasicMarkerFile
from pdart.pipeline.build_browse import BuildBrowse
from pdart.pipeline.build_labels import BuildLabels
from pdart.pipeline.check_downloads import CheckDownloads
from pdart.pipeline.copy_primary_files import CopyPrimaryFiles
from pdart.pipeline.directories import Directories, make_directories
from pdart.pipeline.download_docs import DownloadDocs
from pdart.pipeline.insert_changes import InsertChanges
from pdart.pipeline.make_deliverable import MakeDeliverable
from pdart.pipeline.populate_database import PopulateDatabase
from pdart.pipeline.record_changes import RecordChanges
from pdart.pipeline.reset_pipeline import ResetPipeline
from pdart.pipeline.update_archive import UpdateArchive
from pdart.pipeline.validate_bundle import ValidateBundle
from pdart.logging import PDS_LOGGER


class StateMachine(object):
    """
    This object runs a list of pipeline stages, hardcoded into
    self.stages.  It uses a BasicMarkerFile to show progress and
    record the final state.
    """

    def __init__(self, dirs: Directories, proposal_id: int) -> None:
        self.marker_file = BasicMarkerFile(dirs.working_dir(proposal_id))
        self.stages = [
            ("RESETPIPELINE", ResetPipeline(dirs, proposal_id)),
            ("DOWNLOADDOCS", DownloadDocs(dirs, proposal_id)),
            ("CHECKDOWNLOADS", CheckDownloads(dirs, proposal_id)),
            ("COPYPRIMARYFILES", CopyPrimaryFiles(dirs, proposal_id)),
            ("RECORDCHANGES", RecordChanges(dirs, proposal_id)),
            ("INSERTCHANGES", InsertChanges(dirs, proposal_id)),
            ("POPULATEDATABASE", PopulateDatabase(dirs, proposal_id)),
            ("BUILDBROWSE", BuildBrowse(dirs, proposal_id)),
            ("BUILDLABELS", BuildLabels(dirs, proposal_id)),
            ("UPDATEARCHIVE", UpdateArchive(dirs, proposal_id)),
            ("MAKEDELIVERABLE", MakeDeliverable(dirs, proposal_id)),
            ("VALIDATEBUNDLE", ValidateBundle(dirs, proposal_id)),
        ]

    def next_stage(self, phase: str) -> Optional[Stage]:
        def phase_index() -> int:
            for i, (name, stage) in enumerate(self.stages):
                if name == phase:
                    return i
            raise ValueError(f"unknown phase {phase}.")

        i = phase_index()
        try:
            PDS_LOGGER.log("info", f"{self.stages[i+1][0]}")
            return self.stages[i + 1][1]
        except IndexError:
            return None

    def run(self) -> None:
        self.marker_file.clear_marker()
        stage: Optional[Stage] = self.stages[0][1]
        while stage is not None:
            stage()
            marker_info = self.marker_file.get_marker()
            if marker_info is None:
                raise ValueError(f"{marker_info} is None.")
            if marker_info.state == "SUCCESS":
                stage = self.next_stage(marker_info.phase)
            else:
                stage = None

        # Throw an exception if the machine failed
        marker_info = self.marker_file.get_marker()
        if not marker_info or marker_info.state != "SUCCESS":
            raise RuntimeError("State machine failed.")

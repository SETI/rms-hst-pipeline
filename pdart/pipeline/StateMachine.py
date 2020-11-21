import abc
from typing import Optional

from pdart.pipeline.Stage import Stage
from pdart.pipeline.Directories import Directories
from pdart.pipeline.MarkerFile import BasicMarkerFile
from pdart.pipeline.BuildBrowse import BuildBrowse
from pdart.pipeline.BuildLabels import BuildLabels
from pdart.pipeline.CheckDownloads import CheckDownloads
from pdart.pipeline.CopyPrimaryFiles import CopyPrimaryFiles
from pdart.pipeline.Directories import Directories, make_directories
from pdart.pipeline.DownloadDocs import DownloadDocs
from pdart.pipeline.InsertChanges import InsertChanges
from pdart.pipeline.MakeDeliverable import MakeDeliverable
from pdart.pipeline.PopulateDatabase import PopulateDatabase
from pdart.pipeline.RecordChanges import RecordChanges
from pdart.pipeline.ResetPipeline import ResetPipeline
from pdart.pipeline.UpdateArchive import UpdateArchive
from pdart.pipeline.ValidateBundle import ValidateBundle


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
            assert False, f"unknown phase {phase}"

        i = phase_index()
        try:
            return self.stages[i + 1][1]
        except IndexError:
            return None

    def run(self) -> None:
        self.marker_file.clear_marker()
        stage: Optional[Stage] = self.stages[0][1]
        while stage is not None:
            stage()
            marker_info = self.marker_file.get_marker()
            assert marker_info is not None
            if marker_info.state == "SUCCESS":
                stage = self.next_stage(marker_info.phase)
            else:
                stage = None

        # Throw an exception if the machine failed
        marker_info = self.marker_file.get_marker()
        assert marker_info and marker_info.state == "SUCCESS"

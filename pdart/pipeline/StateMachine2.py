import abc
import fs.path
import os
import os.path
import shutil
from subprocess import CompletedProcess, run
from typing import List, Optional

from pdart.pipeline.BuildBrowse import BuildBrowse
from pdart.pipeline.BuildLabels import BuildLabels
from pdart.pipeline.CopyPrimaryFiles import CopyPrimaryFiles
from pdart.pipeline.Directories import Directories
from pdart.pipeline.InsertChanges import InsertChanges
from pdart.pipeline.MakeDeliverable import MakeDeliverable
from pdart.pipeline.MarkerFile import BasicMarkerFile
from pdart.pipeline.PopulateDatabase import PopulateDatabase
from pdart.pipeline.RecordChanges import RecordChanges
from pdart.pipeline.ResetPipeline import ResetPipeline
from pdart.pipeline.Stage import MarkedStage, Stage
from pdart.pipeline.UpdateArchive import UpdateArchive
from pdart.pipeline.Utils import make_osfs
from pdart.pipeline.ValidateBundle import ValidateBundle


class SaveDownloads(MarkedStage):
    def _run(self) -> None:
        def make_tarball(tarball_name: str, dir_to_compress: str) -> None:
            tarball = fs.path.join(self.working_dir(), tarball_name)
            # Here we create a "hst_NNNNN/xxxDownload.tar.gz" so that
            # if it's expanded where it is, it will re-create the
            # xxxDownload directory.
            cmd: List[str] = [
                "tar",
                "-zcf",
                tarball,
                "-C",  # put relative paths into the archive
                fs.path.dirname(dir_to_compress),
                fs.path.basename(dir_to_compress),
            ]
            completed_process: CompletedProcess = run(cmd)
            completed_process.check_returncode()

        TARBALL_NAMES_AND_DIRS = [
            ("mastDownload.tar.gz", self.mast_downloads_dir()),
            ("documentDownload.tar.gz", self.documents_dir()),
        ]

        for name, dir in TARBALL_NAMES_AND_DIRS:
            make_tarball(name, dir)


class ChangeFiles(MarkedStage):
    def _run(self) -> None:
        def change_fits_file(rel_path: str) -> None:
            abs_path = fs.path.join(
                self.mast_downloads_dir(), fs.path.relpath(rel_path)
            )

            print(f"**** touching {abs_path} ****")
            from TouchFits import touch_fits

            touch_fits(abs_path)

        with make_osfs(self.mast_downloads_dir()) as mast_fs:

            def _change_fits_file() -> None:
                which_file = 0
                for path in mast_fs.walk.files(filter=["*.fits"]):
                    # change only the n-th FITS file then return
                    if which_file == 0:
                        change_fits_file(path)
                        print(f"#### CHANGED {path} ####")
                        return
                    which_file = which_file - 1
                assert False, "fell off the end of change_fits_file in ChangeFiles"

            def _delete_directory() -> None:
                for path in mast_fs.walk.dirs():
                    if len(fs.path.parts(path)) == 3:
                        print(f"#### REMOVED {path} ####")
                        mast_fs.removetree(path)
                        return
                assert False, "fell off the end of delete_directory in ChangeFiles"

            _change_fits_file()
            # _delete_directory()


class ReResetPipeline(MarkedStage):
    """
    A stage in the pipeline for development and debugging.  ****This
    is not intended to be in the final pipeline.****

    We reset the directory by deleting everything except for the
    downloaded document and data files, their archived tarballs, and
    the archive.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        documents_dir: str = self.documents_dir()
        mast_downloads_dir: str = self.mast_downloads_dir()
        archive_dir: str = self.archive_dir()

        if not os.path.isdir(working_dir):
            return
        for entry in os.listdir(working_dir):
            fullpath = os.path.join(working_dir, entry)
            if not (
                fullpath in [documents_dir, mast_downloads_dir, archive_dir]
                or fullpath.endswith(".tar.gz")
                or fullpath.endswith(".db")
            ):
                if os.path.isdir(fullpath):
                    shutil.rmtree(fullpath)
                else:
                    os.unlink(fullpath)
        print(f"&&&& contents of working_dir after re-reset: {os.listdir(working_dir)}")


class StateMachine2(object):
    """
    This object runs a list of pipeline stages, hardcoded into
    self.stages.  It uses a BasicMarkerFile to show progress and
    record the final state.
    """

    def __init__(self, dirs: Directories, proposal_id: int) -> None:
        self.marker_file = BasicMarkerFile(dirs.working_dir(proposal_id))
        self.stages = [
            ("SAVEDOWNLOADS", SaveDownloads(dirs, proposal_id)),
            ("RERESETPIPELINE", ReResetPipeline(dirs, proposal_id)),
            ("CHANGEFILES", ChangeFiles(dirs, proposal_id)),
            # ("DOWNLOADDOCS", DownloadDocs(dirs, proposal_id)),
            # ("CHECKDOWNLOADS", CheckDownloads(dirs, proposal_id)),
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
            print(f"???? {self.stages[i+1][0]} ????")
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

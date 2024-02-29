import os
import os.path
import shutil
from typing import Dict

from fs.base import FS
import fs.copy
from fs.path import abspath, iteratepath
from fs.subfs import SubFS

from pdart.fs.multiversioned.utils import component_files, dirpath_to_lid
from pdart.pds4.lid import LID
from pdart.pds4.lidvid import LIDVID
from pdart.pipeline.changes_dict import (
    CHANGES_DICT_NAME,
    ChangesDict,
    read_changes_dict,
)
from pdart.pipeline.stage import MarkedStage
from pdart.pipeline.utils import (
    make_osfs,
    make_sv_deltas,
    make_sv_osfs,
    make_version_view,
)
from pdart.logging import PDS_LOGGER


def _is_component_path(dirpath: str) -> bool:
    for part in iteratepath(dirpath):
        if "$" not in part:
            return False
    return True


def _merge_primaries(changes_dict: ChangesDict, src_fs: FS, dst_fs: FS) -> None:
    # TODO Not sure that this hits all cases, including removal of
    # files and directories.  Think about it.
    for dirpath in src_fs.walk.dirs(search="depth"):
        if _is_component_path(dirpath):
            lid = dirpath_to_lid(dirpath)
            changed = changes_dict.changed(lid)
            if changed:
                if not dst_fs.isdir(dirpath):
                    dst_fs.makedirs(dirpath)
                src_sub_fs = SubFS(src_fs, dirpath)
                dst_sub_fs = SubFS(dst_fs, dirpath)
                # delete directories in dst that don't exist in src
                for subdirpath in dst_sub_fs.walk.dirs(search="depth"):
                    if not src_sub_fs.isdir(subdirpath):
                        dst_sub_fs.removetree(subdirpath)
                # delete the files in the destination (if any)
                for filepath in component_files(dst_fs, dirpath):
                    dst_sub_fs.remove(filepath)
                # copy the new files across
                src_sub_fs = SubFS(src_fs, dirpath)
                for filepath in component_files(src_fs, dirpath):
                    fs.copy.copy_file(src_sub_fs, filepath, dst_sub_fs, filepath)


class InsertChanges(MarkedStage):
    """
    In this stage, we insert changed files into the primary-deltas
    directory.

    When this stage finishes, for each changed LID, its directory
    should contain only new primary files.  (TODO How do we handle
    subdirectories?  New, changed, and deleted ones?)
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        primary_files_dir: str = self.primary_files_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()
        try:
            PDS_LOGGER.open("Create a directory for a new version of the bundle")
            if os.path.isdir(self.deliverable_dir()):
                raise ValueError(
                    f"{self.deliverable_dir()} cannot exist for InsertChanges."
                )

            changes_path = os.path.join(working_dir, CHANGES_DICT_NAME)
            with make_osfs(archive_dir) as archive_osfs, make_version_view(
                archive_osfs, self._bundle_segment
            ) as version_view, make_sv_osfs(
                primary_files_dir
            ) as primary_files_osfs, make_sv_deltas(
                version_view, archive_primary_deltas_dir
            ) as sv_deltas:
                archive_dirs = list(archive_osfs.walk.dirs())
                changes_dict = read_changes_dict(changes_path)
                _merge_primaries(changes_dict, primary_files_osfs, sv_deltas)

            shutil.rmtree(primary_files_dir + "-sv")
            if not os.path.isdir(archive_dir):
                raise ValueError(f"{archive_dir} doesn't exist.")
            dirpath = archive_primary_deltas_dir + "-deltas-sv"
            PDS_LOGGER.log("info", f"Directory for the new version: {dirpath}")
            if not os.path.isdir(dirpath):
                raise ValueError(f"{dirpath} doesn't exist.")
            if not os.path.isfile(changes_path):
                raise ValueError(f"{changes_path} is not a file.")
        except Exception as e:
            PDS_LOGGER.exception(e)
        finally:
            PDS_LOGGER.close()

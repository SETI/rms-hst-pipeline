import logging
import os.path
from typing import Dict, Iterator, Set

from fs.base import FS
from fs.subfs import SubFS
from fs.path import iteratepath, join, relpath, splitext

from pdart.documents.downloads import DOCUMENT_SUFFIXES
from pdart.fs.multiversioned.multiversioned import Multiversioned
from pdart.fs.multiversioned.utils import dirpath_to_lid
from pdart.pds4.lid import LID
from pdart.pds4.lidvid import LIDVID
from pdart.pds4.vid import VID
from pdart.pipeline.ChangesDict import (
    CHANGES_DICT_NAME,
    ChangesDict,
    write_changes_dict,
)
from pdart.pipeline.utils import (
    make_osfs,
    make_mv_osfs,
    make_sv_osfs,
    make_version_view,
)
from pdart.pipeline.Stage import MarkedStage
from pdart.logging import PDS_LOGGER

_PRIMARY_SUFFIXES = DOCUMENT_SUFFIXES + [".fits", ".txt"]


def _is_primary_file(filepath: str) -> bool:
    _, ext = splitext(filepath)
    return ext in _PRIMARY_SUFFIXES


def _is_primary_dir(dirpath: str) -> bool:
    return "$/browse_" not in dirpath


def _lid_is_primary(lid: LID) -> bool:
    if lid.collection_id:
        return not lid.collection_id.startswith("browse")
    else:
        return True


def _next_vid(mv: Multiversioned, lid: LID, changed: bool) -> VID:
    # TODO If you want to allow minor changes, here is where you
    # decide which VID to use.  Set a test here, then add a
    # parameter that decides between major and minor and thread it
    # up through the call stack.
    latest_lidvid = mv.latest_lidvid(lid)
    if latest_lidvid is None:
        return VID("1.0")
    elif changed:
        return latest_lidvid.vid().next_major_vid()
    else:
        return latest_lidvid.vid()


def _get_primary_changes(
    mv: Multiversioned, primary_fs: FS, latest_version_fs: FS
) -> ChangesDict:
    """
    Walk through the two filesystems, when you find differences
    between the two, if it involves primary files, note it in a
    ChangesDict and return all the changes.

    TODO This function was hacked together.  It works but it's
    possible that the check is quadratic in the size of the tree, not
    linear.  That would be bad.  Review this function and fix it if
    it's wrong.  When done, remove this comment.
    """
    result = ChangesDict()

    def filter_to_primary_files(dir: str, filenames: Iterator[str]) -> Set[str]:
        return {
            filename for filename in filenames if _is_primary_file(join(dir, filename))
        }

    def filter_to_primary_dirs(dir: str, dirnames: Iterator[str]) -> Set[str]:
        return {dirname for dirname in dirnames if _is_primary_dir(join(dir, dirname))}

    def dirs_match(dirpath: str) -> bool:
        primary_dirs = filter_to_primary_dirs(
            dirpath,
            (
                relpath(dir)
                for dir in SubFS(primary_fs, dirpath).walk.dirs()
                if "$" in dir
            ),
        )
        latest_dirs = filter_to_primary_dirs(
            dirpath,
            (
                relpath(dir)
                for dir in SubFS(latest_version_fs, dirpath).walk.dirs()
                if "$" in dir
            ),
        )
        PDS_LOGGER.open("Directory changes detected")
        if primary_dirs == latest_dirs:
            for dir in primary_dirs:
                full_dirpath = join(dirpath, relpath(dir))
                lid = dirpath_to_lid(full_dirpath)
                if lid not in result.changes_dict:
                    raise KeyError(f"{lid} not in changes_dict.")
                if result.changed(lid):
                    PDS_LOGGER.log(
                        "info", f"CHANGE DETECTED in {dirpath}: {lid} changed"
                    )
                    PDS_LOGGER.close()
                    return False
            PDS_LOGGER.close()
            return True
        else:
            # list of dirs does not match
            added = primary_dirs - latest_dirs
            removed = latest_dirs - primary_dirs
            if added and removed:
                PDS_LOGGER.log(
                    "info",
                    f"CHANGE DETECTED IN {dirpath}: added {added}; removed {removed}",
                )
            elif added:
                PDS_LOGGER.log("info", f"CHANGE DETECTED IN {dirpath}: added {added}")
            else:  # removed
                PDS_LOGGER.log(
                    "info", f"CHANGE DETECTED IN {dirpath}: removed {removed}"
                )
            PDS_LOGGER.close()
            return False

    def files_match(dirpath: str) -> bool:
        # All files in subcomponents will have a "$" in their path (it
        # comes after the name of the subcomponent), so by filtering
        # them out, we get only the files for this component.  PDS4
        # *does* allow directories in a component (that aren't part of
        # a subcomponent), so we use walk instead of listdir() to get
        # *all* the files, not just the top-level ones.
        primary_files = filter_to_primary_files(
            dirpath,
            (
                relpath(filepath)
                for filepath in SubFS(primary_fs, dirpath).walk.files()
                if "$" not in filepath
            ),
        )
        latest_files = filter_to_primary_files(
            dirpath,
            (
                relpath(filepath)
                for filepath in SubFS(latest_version_fs, dirpath).walk.files()
                if "$" not in filepath
            ),
        )
        PDS_LOGGER.open("File changes detected")
        if primary_files != latest_files:
            PDS_LOGGER.log(
                "info",
                f"CHANGE DETECTED IN {dirpath}: {primary_files} != {latest_files}",
            )
            PDS_LOGGER.close()
            return False
        for filename in primary_files:
            filepath = join(dirpath, relpath(filename))
            if primary_fs.getbytes(filepath) != latest_version_fs.getbytes(filepath):
                PDS_LOGGER.log(
                    "info", f"CHANGE DETECTED IN {filepath}; DIRPATH = {dirpath}"
                )
                PDS_LOGGER.close()
                return False
        PDS_LOGGER.close()
        return True

    for dirpath in primary_fs.walk.dirs(filter=["*\$$"], search="depth"):
        lid = dirpath_to_lid(dirpath)

        if _lid_is_primary(lid):
            latest_lidvid = mv.latest_lidvid(lid)
            if latest_version_fs.isdir(dirpath):
                matches = files_match(dirpath) and dirs_match(dirpath)
                result.set(lid, _next_vid(mv, lid, not matches), not matches)
            else:
                result.set(lid, _next_vid(mv, lid, True), True)
        else:
            pass
    return result


class RecordChanges(MarkedStage):
    """
    We compare the downloaded files with the latest versions in the
    archive.  We make a list of the LIDVIDs that have changed and
    write them into the CHANGES_DICT_NAME.

    Note that when we have a technique to tell which files on MAST
    have changed, so we can download only the changed files, then this
    stage will not be needed.

    When this stage finishes, there should (still) be a
    primary_files_dir, but we have added a CHANGES_DICT_NAME.
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        primary_files_dir: str = self.primary_files_dir()
        archive_dir: str = self.archive_dir()

        if os.path.isdir(self.deliverable_dir()):
            raise ValueError(
                f"{self.deliverable_dir()} cannot exist " + "for RecordChanges"
            )

        if not os.path.isdir(working_dir):
            raise ValueError(f"{working_dir} doesn't exist.")
        if not os.path.isdir(primary_files_dir + "-sv"):
            raise ValueError(f"{primary_files_dir}-sv doesn't exist.")

        changes: Dict[LIDVID, bool] = dict()
        changes_path = os.path.join(working_dir, CHANGES_DICT_NAME)
        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as latest_version:
            with make_sv_osfs(primary_files_dir) as primary_fs:
                mv = Multiversioned(archive_osfs)
                d = _get_primary_changes(mv, primary_fs, latest_version)
                write_changes_dict(d, changes_path)

                if not d.has_changes():
                    print("#### PRIMARY_FS ################")
                    primary_fs.tree()

                    print("#### LATEST_VERSION ################")
                    latest_version.tree()

        if not os.path.isdir(primary_files_dir + "-sv"):
            raise ValueError(f"{primary_files_dir}-sv doesn't exist.")
        if not os.path.isfile(os.path.join(working_dir, CHANGES_DICT_NAME)):
            raise ValueError(
                f"{os.path.join(working_dir, CHANGES_DICT_NAME)} " + "doesn't exist."
            )

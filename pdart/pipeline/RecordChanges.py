from typing import Dict, Iterator, Set
import os.path

from fs.base import FS
from fs.subfs import SubFS
from fs.path import iteratepath, join, relpath, splitext

from pdart.fs.multiversioned.Multiversioned import Multiversioned
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.ChangesDict import (
    CHANGES_DICT_NAME,
    ChangesDict,
    write_changes_dict,
)
from pdart.pipeline.Utils import (
    make_osfs,
    make_mv_osfs,
    make_sv_osfs,
    make_version_view,
)
from pdart.pipeline.Stage import MarkedStage


def _is_primary_file(filepath: str) -> bool:
    PRIMARY_SUFFIXES = [".fits", ".apt", ".pdf", ".pro", ".prop", ".txt"]
    _, ext = splitext(filepath)
    return ext in PRIMARY_SUFFIXES


def _is_primary_dir(dirpath: str) -> bool:
    return "$/browse_" not in dirpath


def dir_to_lid(dir: str) -> LID:
    """
    Convert a directory path to a LID.  Raise on errors.
    """
    parts = [str(part[:-1]) for part in iteratepath(dir) if "$" in part]
    return LID.create_from_parts(parts)


def _lid_is_primary(lid: LID) -> bool:
    if lid.collection_id:
        return not lid.collection_id.startswith("browse")
    else:
        return True


def _get_primary_changes(
    mv: Multiversioned, primary_fs: FS, latest_version_fs: FS
) -> ChangesDict:
    result = ChangesDict()

    def filter_to_primary_files(filenames: Iterator[str]) -> Set[str]:
        return {filename for filename in filenames if _is_primary_file(filename)}

    def filter_to_primary_dirs(dirnames: Iterator[str]) -> Set[str]:
        return {dirname for dirname in dirnames if _is_primary_dir(dirname)}

    def dirs_match(dirpath: str) -> bool:
        primary_dirs = filter_to_primary_dirs(
            dir for dir in SubFS(primary_fs, dirpath).walk.dirs() if "$" in dir
        )
        latest_dirs = filter_to_primary_dirs(
            dir for dir in SubFS(primary_fs, dirpath).walk.dirs() if "$" in dir
        )
        if primary_dirs == latest_dirs:
            for dir in primary_dirs:
                full_dirpath = join(dirpath, relpath(dir))
                lid = dir_to_lid(full_dirpath)
                assert lid in result.changes_dict
                if result.changed(lid):
                    return False
            return True
        else:
            # list of dirs does not match
            return False

    def files_match(dirpath: str) -> bool:
        primary_files = filter_to_primary_files(
            filepath
            for filepath in SubFS(primary_fs, dirpath).walk.files()
            if "$" not in filepath
        )
        latest_files = filter_to_primary_files(
            filepath
            for filepath in SubFS(latest_version_fs, dirpath).walk.files()
            if "$" not in filepath
        )
        if primary_files != latest_files:
            return False
        for filename in primary_files:
            filepath = join(dirpath, relpath(filename))
            if primary_fs.getbytes(filepath) != latest_version_fs.getbytes(filepath):
                print(f"#### CHANGE DETECTED IN {filepath}; DIRPATH = {dirpath} ####")
                return False
        return True

    def next_vid(lid: LID, changed: bool) -> VID:
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

    for dirpath in primary_fs.walk.dirs(filter=["*\$$"], search="depth"):
        lid = dir_to_lid(dirpath)

        if _lid_is_primary(lid):
            latest_lidvid = mv.latest_lidvid(lid)
            if latest_version_fs.isdir(dirpath):
                matches = files_match(dirpath) and dirs_match(dirpath)
                result.set(lid, next_vid(lid, not matches), not matches)
            else:
                result.set(lid, next_vid(lid, True), True)
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

        assert not os.path.isdir(
            self.deliverable_dir()
        ), "{deliverable_dir} cannot exist for RecordChanges"

        assert os.path.isdir(working_dir), working_dir
        assert os.path.isdir(primary_files_dir + "-sv"), primary_files_dir

        changes: Dict[LIDVID, bool] = dict()
        changes_path = os.path.join(working_dir, CHANGES_DICT_NAME)
        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as latest_version:
            with make_sv_osfs(primary_files_dir) as primary_fs:
                mv = Multiversioned(archive_osfs)
                d = _get_primary_changes(mv, primary_fs, latest_version)
                write_changes_dict(d, changes_path)
                d.dump("after RecordChanges")
                print(
                    "----------------------------------------------------------------"
                )

        assert os.path.isdir(primary_files_dir + "-sv")
        assert os.path.isfile(os.path.join(working_dir, CHANGES_DICT_NAME))

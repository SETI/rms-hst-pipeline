import os.path
from typing import Dict

from fs.path import iteratepath

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Utils import make_mv_osfs, make_sv_osfs, make_version_view
from pdart.pipeline.Stage import Stage

CHANGES_DICT: str = "changes$dict.txt"


def dir_to_lid(dir: str) -> LID:
    """
    Convert a directory path to a LID.  Raise on errors.
    """
    parts = [str(part[:-1]) for part in iteratepath(dir) if "$" in part]
    return LID.create_from_parts(parts)


class RecordChanges(Stage):
    def _run(self) -> None:
        working_dir: str = self.working_dir()
        primary_files_dir: str = self.primary_files_dir()
        archive_dir: str = self.archive_dir()

        assert os.path.isdir(working_dir), working_dir
        assert os.path.isdir(primary_files_dir + "-sv"), primary_files_dir

        changes: Dict[LIDVID, bool] = dict()
        changes_path = os.path.join(working_dir, CHANGES_DICT)
        if os.path.isdir(archive_dir):
            # TODO
            with make_mv_osfs(archive_dir) as archive_osfs, make_version_view(
                archive_osfs, self._bundle_segment
            ) as latest_version:
                assert (
                    False
                ), "record_changes for existing archive not fully implemented"
        else:
            # There is no archive, so all the LIDVIDs are new.
            vid = VID("1.0")
            with make_sv_osfs(primary_files_dir) as osfs:
                with open(changes_path, "w") as changes_file:
                    for dir in osfs.walk.dirs():
                        lid = dir_to_lid(dir)
                        lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
                        print(lidvid, "True", file=changes_file)

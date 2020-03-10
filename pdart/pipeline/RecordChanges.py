from typing import TYPE_CHECKING
import os.path
import pickle

from fs.path import iteratepath
from multiversioned.VersionView import VersionView
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pipeline.Utils import make_mv_osfs, make_sv_osfs, make_version_view

if TYPE_CHECKING:
    from typing import Dict
    from pdart.pds4.LIDVID import LIDVID

CHANGES_DICT = "changes$dict.txt"


def dir_to_lid(dir):
    # type: (unicode) -> LID
    """
    Convert a directory path to a LID.  Raise on errors.
    """
    parts = [str(part[:-1]) for part in iteratepath(dir) if "$" in part]
    return LID.create_from_parts(parts)


def record_changes(bundle_segment, working_dir, primary_files_dir, archive_dir):
    # type: (unicode, unicode, unicode, unicode) -> None
    assert os.path.isdir(working_dir), working_dir
    assert os.path.isdir(primary_files_dir + "-sv"), primary_files_dir

    changes = dict()  # type: Dict[LIDVID, bool]
    changes_path = os.path.join(working_dir, CHANGES_DICT)
    if os.path.isdir(archive_dir):
        # TODO
        with make_mv_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, bundle_segment
        ) as latest_version:
            assert False, "record_changes for existing archive not fully implemented"
    else:
        # There is no archive, so all the LIDVIDs are new.
        vid = VID("1.0")
        with make_sv_osfs(primary_files_dir) as osfs:
            with open(changes_path, "w") as changes_file:
                for dir in osfs.walk.dirs():
                    lid = dir_to_lid(dir)
                    lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
                    print >> changes_file, lidvid, "True"

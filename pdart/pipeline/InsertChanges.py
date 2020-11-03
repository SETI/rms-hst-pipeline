import os
import os.path
import shutil
from typing import Dict

import fs.copy

from pdart.pds4.LIDVID import LIDVID
from pdart.pipeline.RecordChanges import CHANGES_DICT
from pdart.pipeline.Stage import MarkedStage
from pdart.pipeline.Utils import (
    make_osfs,
    make_sv_deltas,
    make_sv_osfs,
    make_version_view,
)


def read_changes_dict(changes_path: str) -> Dict[LIDVID, str]:
    changes_dict = dict()
    with open(changes_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                assert len(parts) == 2, parts
                lidvid, changed = parts
                changes_dict[LIDVID(lidvid)] = changed
    return changes_dict


class InsertChanges(MarkedStage):
    """
    In this stage, we insert changed files into the primary-deltas
    directory.  (The CHANGES_DICT seems to be unused.  TODO Verify and
    if so, remove it.)

    TODO The insertion is currently unimplemented except in the case
    that this is the first version and so the archive is empty.

    TODO When this stage finishes, there should be...what?
    """

    def _run(self) -> None:
        working_dir: str = self.working_dir()
        primary_files_dir: str = self.primary_files_dir()
        archive_dir: str = self.archive_dir()
        archive_primary_deltas_dir: str = self.archive_primary_deltas_dir()

        changes_path = os.path.join(working_dir, CHANGES_DICT)
        with make_osfs(archive_dir) as archive_osfs, make_version_view(
            archive_osfs, self._bundle_segment
        ) as version_view, make_sv_osfs(
            primary_files_dir
        ) as primary_files_osfs, make_sv_deltas(
            version_view, archive_primary_deltas_dir
        ) as sv_deltas:
            archive_dirs = list(archive_osfs.walk.dirs())
            if archive_dirs:
                changes_dict = read_changes_dict(changes_path)
                # TODO write a merge algorithm
                assert False, "need an algorithm to merge changes into archive"
            else:
                # the archive is empty and we can just copy into it
                for dirpath in primary_files_osfs.walk.dirs():
                    sv_deltas.makedirs(dirpath)
                for filepath in primary_files_osfs.walk.files():
                    fs.copy.copy_file(primary_files_osfs, filepath, sv_deltas, filepath)

        shutil.rmtree(primary_files_dir + "-sv")

        assert os.path.isdir(archive_dir), archive_dir
        dirpath = archive_primary_deltas_dir + "-deltas-sv"
        assert os.path.isdir(dirpath), dirpath
        assert os.path.isfile(changes_path)

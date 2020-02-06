from contextlib import closing
import os
import os.path
import shutil

from fs.osfs import OSFS

from cowfs.COWFS import COWFS
from multiversioned.Multiversioned import Multiversioned, std_is_new
from pdart.pds4.LID import LID
from pdart.pipeline.Utils import *


def make_new_versions(bundle_segment,
                      archive_dir,
                      next_version_deltas_dir,
                      archive_next_version_deltas_dir):
    # type: (unicode, unicode, unicode, unicode) -> None
    assert os.path.isdir(archive_dir)
    assert os.path.isdir(next_version_deltas_dir)

    with make_osfs(archive_dir) as last_archive_fs, \
            make_version_view(last_archive_fs,
                              bundle_segment) as last_version_view_fs, \
            make_deltas(last_version_view_fs,
                        next_version_deltas_dir) as next_version_view_fs, \
            make_deltas(last_archive_fs,
                        archive_next_version_deltas_dir) \
                        as archive_next_version_fs:

        show_tree('last_archive_fs', last_archive_fs)
        show_tree('last_version_view_fs', last_version_view_fs)
        show_tree('next_version_view_fs', next_version_view_fs)
        show_tree('archive_next_version_fs', archive_next_version_fs)

        mv = Multiversioned(archive_next_version_fs)
        changed = mv.update_from_single_version(std_is_new,
                                                next_version_view_fs)
        show_tree('archive_next_version_fs', archive_next_version_fs)

    shutil.rmtree(next_version_deltas_dir)
    if changed:
        assert os.path.isdir(archive_next_version_deltas_dir)
    else:
        shutil.rmtree(archive_next_version_deltas_dir)
        assert not os.path.isdir(archive_next_version_deltas_dir)

    assert os.path.isdir(archive_dir)
    assert not os.path.isdir(next_version_deltas_dir)


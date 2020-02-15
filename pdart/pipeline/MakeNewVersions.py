from contextlib import closing
import os
import os.path
import shutil

from fs.osfs import OSFS

from cowfs.COWFS import COWFS
from multiversioned.Multiversioned import Multiversioned, std_is_new
from pdart.pds4.LID import LID
from pdart.pipeline.Utils import *

_VERBOSE = False

def make_new_versions(bundle_segment,
                      archive_dir,
                      next_version_deltas_dir,
                      archive_next_version_fits_and_docs_dir):
    # type: (unicode, unicode, unicode, unicode) -> None
    assert os.path.isdir(archive_dir + '-mv')
    assert os.path.isdir(next_version_deltas_dir + '-deltas-sv')

    with make_mv_osfs(archive_dir) as last_archive_fs, \
            make_version_view(last_archive_fs,
                              bundle_segment) as last_version_view_fs, \
            make_sv_deltas(last_version_view_fs,
                        next_version_deltas_dir) as next_version_view_fs, \
            make_mv_deltas(last_archive_fs,
                           archive_next_version_fits_and_docs_dir) \
                           as archive_next_version_fits_and_docs_fs:

        if _VERBOSE:
            show_tree('last_archive_fs', last_archive_fs)
            show_tree('last_version_view_fs', last_version_view_fs)
            show_tree('next_version_view_fs', next_version_view_fs)
            show_tree('archive_next_version_fits_and_docs_fs',
                      archive_next_version_fits_and_docs_fs)

        mv = Multiversioned(archive_next_version_fits_and_docs_fs)
        changed = mv.update_from_single_version(std_is_new,
                                                next_version_view_fs)
        if _VERBOSE:
            show_tree('archive_next_version_fits_and_docs_fs',
                      archive_next_version_fits_and_docs_fs)

    shutil.rmtree(next_version_deltas_dir + '-deltas-sv')
    if changed:
        assert os.path.isdir(archive_next_version_fits_and_docs_dir + '-deltas-mv')
    else:
        shutil.rmtree(archive_next_version_fits_and_docs_dir + '-deltas-mv')
        assert not os.path.isdir(archive_next_version_fits_and_docs_dir + 
                                 '-deltas-mv')

    assert os.path.isdir(archive_dir + '-mv')
    assert not os.path.isdir(next_version_deltas_dir + '-deltas-sv')


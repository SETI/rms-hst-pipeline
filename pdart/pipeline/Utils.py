from typing import TYPE_CHECKING
from contextlib import contextmanager
import os.path
import shutil

from multiversioned.Multiversioned import Multiversioned

from cowfs.COWFS import COWFS
from fs.osfs import OSFS
from fs.tempfs import TempFS
import fs.path

from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID

from pdart.pipeline.FSTypes import *

if TYPE_CHECKING:
    from typing import Generator
    from fs.base import FS
    from multiversioned.VersionView import VersionView

def show_tree(tag, fs):
    # type: (str, FS) -> None
    line = '---- %s ' % tag
    tag_len = len(tag)
    line += (60 - tag_len) * '-'
    print line
    fs.tree()

@contextmanager
def make_osfs(dir):
    # type: (unicode) -> Generator[OSFS, None, None]
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fs = OSFS(dir)
    yield fs
    fs.close()

@contextmanager
def make_deltas(base_fs, cow_dirpath):
    # type: (FS, unicode) -> Generator[COWFS, None, None]
    with make_osfs(cow_dirpath) as osfs:
        fs = COWFS.create_cowfs(base_fs, osfs, True)
        cat_base_fs = categorize_filesystem(base_fs)
        cat_fs = categorize_filesystem(fs)
        if cat_base_fs != EMPTY_FS_TYPE  and \
                cat_fs != EMPTY_FS_TYPE and \
                cat_fs != cat_base_fs:
            print ('%s, %s' % (cat_base_fs, cat_fs))
            show_tree('base_fs', base_fs)
            show_tree('fs', fs)
            assert False
        yield fs
        fs.close()

@contextmanager
def make_version_view(archive_osfs, bundle_segment):
    # type: (OSFS, unicode) -> Generator[VersionView, None, None]
    assert categorize_filesystem(archive_osfs) \
        in [EMPTY_FS_TYPE, MULTIVERSIONED_FS_TYPE], \
        categorize_filesystem(archive_osfs)
    mv = Multiversioned(archive_osfs)
    lid = LID('urn:nasa:pds:' + str(bundle_segment))
    res = mv.create_version_view(lid)
    assert categorize_filesystem(res) \
        in [EMPTY_FS_TYPE, SINGLE_VERSIONED_FS_TYPE], \
        categorize_filesystem(res)

    yield res
    res.close()


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
        yield fs
        fs.close()

@contextmanager
def make_version_view(archive_osfs, bundle_segment):
    # type: (OSFS, unicode) -> Generator[VersionView, None, None]
    mv = Multiversioned(archive_osfs)
    lid = LID('urn:nasa:pds:' + str(bundle_segment))
    res = mv.create_version_view(lid)
    yield res
    res.close()


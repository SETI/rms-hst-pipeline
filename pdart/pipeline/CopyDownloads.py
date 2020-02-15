from typing import TYPE_CHECKING
from contextlib import contextmanager
import os.path
import shutil

from multiversioned.Multiversioned import Multiversioned
from pdart.pipeline.Utils import *

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

def to_segment_dir(name):
    return name + '$'


def copy_downloads(bundle_segment, mast_downloads_dir,
                   fits_and_docs_deltas_dir, archive_dir):
    # type: (unicode, unicode, unicode, unicode) -> None
    assert os.path.isdir(mast_downloads_dir)

    assert bundle_segment.startswith('hst_')
    assert bundle_segment[-5:].isdigit()

    with make_osfs(mast_downloads_dir) as mast_downloads_fs, \
        make_mv_osfs(archive_dir) as archive_fs, \
        make_version_view(archive_fs, bundle_segment) as version_fs, \
        make_sv_deltas(version_fs,
                       fits_and_docs_deltas_dir) \
                       as fits_and_docs_delta_fs:

        # Walk the mast_downloads_dir for FITS file and file
        # them into the COW filesystem.
        for filepath in mast_downloads_fs.walk.files(filter=['*.fits']):
            parts = fs.path.iteratepath(filepath)
            depth = len(parts)
            assert depth == 3, parts
            _, product, filename = parts
            filename = filename.lower()
            hst_filename = HstFilename(filename)
            coll = 'data_%s_%s' % (hst_filename.instrument_name(),
                                   hst_filename.suffix())
            new_path = fs.path.join(to_segment_dir(bundle_segment),
                                    to_segment_dir(coll),
                                    to_segment_dir(product), filename)
            dirs, filename = fs.path.split(new_path)
            fits_and_docs_delta_fs.makedirs(dirs)
            fs.copy.copy_file(mast_downloads_fs, filepath,
                              fits_and_docs_delta_fs, new_path)

    assert os.path.isdir(archive_dir + '-mv'), (archive_dir + '-mv')
    assert os.path.isdir(fits_and_docs_deltas_dir + '-deltas-sv'), \
        (fits_and_docs_deltas_dir + '-deltas-sv')
    # If I made it to here, it should be safe to delete the downloads
    shutil.rmtree(mast_downloads_dir)
    assert not os.path.isdir(mast_downloads_dir)

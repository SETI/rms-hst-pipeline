import os
import shutil

import fs.copy
import fs.path
from fs.osfs import OSFS

from pdart.fs.DeliverableFS import DeliverableFS
from pdart.fs.V1FS import V1FS
from pdart.pds4.HstFilename import HstFilename


# Work on rewriting pdart.archive.StartArchive but using the
# pyfilesystem to shuffle directories around.


def start_archive(src_dir, dst_dir):
    # type: (unicode, unicode) -> None
    src_fs = OSFS(src_dir)
    print 'Source looks like this:'
    src_fs.tree()
    dst_fs = V1FS(dst_dir)
    for filepath in src_fs.walk.files(filter=['*.fits']):
        parts = fs.path.iteratepath(filepath)
        depth = len(parts)
        assert depth == 5, filepath
        bundle, _, _, product, filename = parts
        filename = filename.lower()
        hst_filename = HstFilename(filename)
        coll = 'data_%s_%s' % (hst_filename.instrument_name(),
                               hst_filename.suffix())
        new_path = fs.path.join(bundle, coll, product, filename)
        dirs, filename = fs.path.split(new_path)
        dst_fs.makedirs(dirs)
        fs.copy.copy_file(src_fs, filepath, dst_fs, new_path)
    print 'Archive, seen as uni-versioned, looks like this:'
    dst_fs.tree()
    print 'Archive, seen as multi-versioned, looks like this:'
    OSFS(dst_dir).tree()


def make_delivery(src_dir, dst_dir):
    src_fs = V1FS(src_dir)
    print '===='
    osfs = OSFS(dst_dir)
    dst_fs = DeliverableFS(osfs)
    fs.copy.copy_fs(src_fs, dst_fs)
    print 'Delivery looks like this:'
    osfs.tree()


if __name__ == '__main__':
    shutil.rmtree(u'/Users/spaceman/bd-archive')
    os.mkdir(u'/Users/spaceman/bd-archive')
    shutil.rmtree(u'/Users/spaceman/bd-deliverable')
    os.mkdir(u'/Users/spaceman/bd-deliverable')

    start_archive(u'/Users/spaceman/bulk-download',
                  u'/Users/spaceman/bd-archive')
    make_delivery(u'/Users/spaceman/bd-archive',
                  u'/Users/spaceman/bd-deliverable')

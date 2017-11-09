"""
Preliminary work on introducing versioning into the system.  First step is
to reorganize files into the new format.
"""
import shutil
from os import makedirs

from fs.copy import copy_fs
from fs.osfs import OSFS
from fs.path import join, normpath

from pdart.fs.InitialVersionedView import InitialVersionedView
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.pds4.Archive import Archive
from pdart.pds4.Archives import get_any_archive_dir

LOGGING = True


def on_copy(src_fs, src_path, dst_fs, dst_path):
    print ('copying to ' + dst_path)


RECREATE = True

if __name__ == '__main__':
    # This copies a set of bundles in old format (src) to a directory
    # in the new format (dst).
    src = get_any_archive_dir()
    dst = normpath(join(src, '..', 'NewArchive'))

    if RECREATE:
        shutil.rmtree(dst)
        makedirs(dst)

    for bundle in Archive(src).bundles():
        bundle_id = bundle.lid.bundle_id

        initBundle = InitialVersionedView(bundle_id,
                                          OSFS(bundle.absolute_filepath()))
        newBundle = OSFS(dst, create=True)

        if LOGGING:
            copy_fs(initBundle, newBundle, walker=None, on_copy=on_copy)
        else:
            copy_fs(initBundle, newBundle)

    if LOGGING:
        mvb_fs = MultiversionBundleFS(OSFS(dst))
        mvb_fs.tree()

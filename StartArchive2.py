import os
import os.path
import shutil

import fs.copy
import fs.path
from fs.osfs import OSFS
from fs.tarfs import TarFS

from pdart.archive.StartBundle import start_bundle
from pdart.fs.DeliverableFS import DeliverableFS
from pdart.fs.V1FS import V1FS
from pdart.pds4.HstFilename import HstFilename


# Work on rewriting pdart.archive.StartArchive but using the
# pyfilesystem to shuffle directories around.
def is_bundle_name(name):
    return name.startswith('hst_')


def start_archive(src_dir, dst_dir, tar_dir):
    # type: (unicode, unicode, unicode) -> None
    bundle_names = [dirname for dirname in os.listdir(src_dir)
                    if is_bundle_name(dirname)
                    and os.path.isdir(os.path.join(src_dir, dirname))]
    # TODO Remove limit.
    for bundle_name in bundle_names[0:5]:
        dst_bundle_dir = fs.path.join(dst_dir, bundle_name)
        os.makedirs(dst_bundle_dir)
        start_bundle(src_dir, bundle_name, dst_bundle_dir)
        t_src_dir = dst_bundle_dir
        t_dst_dir = fs.path.join(tar_dir, bundle_name)
        src_fs = V1FS(t_src_dir)
        os.makedirs(t_dst_dir)
        dst_fs = OSFS(t_dst_dir)
        dst_del_fs = DeliverableFS(dst_fs)
        fs.copy.copy_fs(src_fs, dst_del_fs)
        # TODO add manifests to dst_fs (not dst_del_fs)
        tarfile_name = '%s.tar.gz' % bundle_name
        with TarFS(fs.path.join(tar_dir, tarfile_name), write=True) as tar_fs:
            fs.copy.copy_fs(dst_fs, tar_fs)


def make_delivery(src_dir, dst_dir):
    src_fs = V1FS(src_dir)
    print '===='
    osfs = OSFS(dst_dir)
    dst_fs = DeliverableFS(osfs)
    fs.copy.copy_fs(src_fs, dst_fs)
    print 'Delivery looks like this:'
    osfs.tree()


if __name__ == '__main__':
    SRC_DIR = '/Volumes/PDART-5TB Part Deux/bulk-download/'
    DST_DIR = '/Volumes/PDART-8TB/archive'
    TAR_DIR = '/Volumes/PDART-8TB/tarfiles'

    if True:
        shutil.rmtree(DST_DIR, True)
        os.makedirs(DST_DIR)
        shutil.rmtree(TAR_DIR, True)
        os.makedirs(TAR_DIR)

    start_archive(SRC_DIR, DST_DIR, TAR_DIR)

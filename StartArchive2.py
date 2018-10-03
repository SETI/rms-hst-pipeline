import os
import os.path
import shutil

import fs.copy
import fs.path
from fs.osfs import OSFS
from fs.tarfs import TarFS

from pdart.archive.ChecksumManifest import make_checksum_manifest
from pdart.archive.TransferManifest import make_transfer_manifest
from pdart.archive.StartBundle import start_bundle, _BUNDLE_DB_NAME
from pdart.fs.DeliverableFS import DeliverableFS, lidvid_to_dirpath
from pdart.fs.V1FS import V1FS
from pdart.new_db.BundleDB import create_bundle_db_from_os_filepath
from pdart.new_db.Utils import file_md5
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
    for bundle_name in bundle_names:
        try:
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
            os_filepath = fs.path.join(DST_DIR, bundle_name,
                                       bundle_name, _BUNDLE_DB_NAME)
            bundle_db = create_bundle_db_from_os_filepath(os_filepath)
            checksum_manifest = make_checksum_manifest(bundle_db,
                                                       lidvid_to_dirpath)
            filepath = fs.path.join(tar_dir, bundle_name,
                                    'checksum.manifest.txt')
            with open(filepath, 'w') as f:
                f.write(checksum_manifest)

            transfer_manifest = make_transfer_manifest(bundle_db,
                                                       lidvid_to_dirpath)

            filepath = fs.path.join(tar_dir, bundle_name,
                                    'transfer.manifest.txt')
            with open(filepath, 'w') as f:
                f.write(transfer_manifest)

            # make the tarfile
            tarfile_name = '%s.tar.gz' % bundle_name
            tarfile_path = fs.path.join(tar_dir, bundle_name, tarfile_name)
            with TarFS(tarfile_path, write=True) as tar_fs:
                fs.copy.copy_fs(dst_fs, tar_fs)

            # delete the old dir
            if True:
                shutil.rmtree(fs.path.join(t_dst_dir, bundle_name))

            # hash
            hash_filepath = fs.path.join(tar_dir, bundle_name,
                                         '%s.md5' % tarfile_name)
            with open(hash_filepath, 'w') as f:
                f.write(file_md5(tarfile_path))
        except Exception as e:
            print 'FAILURE on bundle %s: %s' % (bundle_name, repr(e))


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

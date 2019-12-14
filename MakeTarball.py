from contextlib import contextmanager
import os
import shutil
import sys
import tempfile

import fs.copy
import fs.path
from fs.osfs import OSFS
from fs.tarfs import TarFS

from pdart.archive.StartBundle import start_bundle
from pdart.fs.DeliverableFS import DeliverableFS
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView
from pdart.pds4.LIDVID import LIDVID


@contextmanager
def temp_directory():
    d = tempfile.mkdtemp(prefix='make_tarball_tmp')
    yield d
    shutil.rmtree(d, True
)

def clean_dir(path):
    shutil.rmtree(path, True)
    os.mkdir(path)


def make_single_tarball(temp_fs=None):
    clean_dir('/Users/spaceman/pdart/new-init-bundle')

    print 'starting the bundle'
    start_bundle('/Users/spaceman/pdart/new-bulk-download',
                 'hst_11187',
                 '/Users/spaceman/pdart/new-init-bundle')

    print 'finished the bundle'
    print 'creating deliverable'

    lidvid = LIDVID('urn:nasa:pds:hst_11187::1.0')
    src = VersionView(
        lidvid,
        MultiversionBundleFS(
            OSFS('/Users/spaceman/pdart/new-init-bundle')))

    clean_dir('/Users/spaceman/pdart/new-tar-site')
    tar_fs = OSFS('/Users/spaceman/pdart/new-tar-site')
    dfs = DeliverableFS(tar_fs)
    fs.copy.copy_fs(src, dfs)
    print 'created deliverable'
    print 'creating tarball'

    if temp_fs:
        with TarFS('hst_11187-1.0.tar.gz', write=True, temp_fs=temp_fs) as t:
            fs.copy.copy_fs(tar_fs, t)
    else:
        with TarFS('hst_11187-1.0.tar.gz', write=True) as t:
            fs.copy.copy_fs(tar_fs, t)

    print 'created tarball'

def make_one_of_many_tarballs(src_dir, dst_dir, proposal_id, temp_fs=None):
    with temp_directory() as init_bundle_dir:
        bundle_name = 'hst_%05d' % proposal_id 
        print 'starting the bundle %s' % bundle_name
        start_bundle(src_dir,
                     bundle_name,
                     init_bundle_dir)

        print 'finished the bundle %s' % bundle_name
        print 'creating deliverable for %s' % bundle_name

        lidvid = LIDVID('urn:nasa:pds:hst_%05d::1.0' % proposal_id)
        src = VersionView(
            lidvid,
            MultiversionBundleFS(
                OSFS(init_bundle_dir)))

        with temp_directory() as tar_dir:
            tar_fs = OSFS(tar_dir)
            dfs = DeliverableFS(tar_fs)
            fs.copy.copy_fs(src, dfs)
            print 'created deliverable for %s' % bundle_name
            print 'creating tarball for %s' % bundle_name
            target_filepath = fs.path.join(dst_dir,
                                           'hst_%05d-1.0.tar.gz' % proposal_id)
            # TODO temp_fs is what you  need
            # with TarFS(target_filepath,
            #            write=True, temp_fs=None) as t:
            with TarFS(target_filepath, write=True) as t:
                fs.copy.copy_fs(tar_fs, t)
            print 'created tarball for %s' % bundle_name

def usage():
    print 'usage: python MakeTarball.py <download-dir> <tarfile-dir> <proposal-id>'
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()

    download_dir, tarfile_dir, proposal_id_str = sys.argv[1:]
    try:
        proposal_id = int(proposal_id_str)
    except ValueError:
        usage()

    make_one_of_many_tarballs(download_dir,
                              tarfile_dir,
                              proposal_id)

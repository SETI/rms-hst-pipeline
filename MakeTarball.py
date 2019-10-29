import os
import shutil

import fs.copy
from fs.osfs import OSFS
from fs.tarfs import TarFS

from pdart.archive.StartBundle import start_bundle
from pdart.fs.DeliverableFS import DeliverableFS
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView
from pdart.pds4.LIDVID import LIDVID


def clean_dir(path):
    shutil.rmtree(path, True)
    os.mkdir(path)


if __name__ == '__main__':
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
    with TarFS('hst_11187-1.0.tar.gz', write=True) as t:
        fs.copy.copy_fs(tar_fs, t)
    print 'created tarball'

import os
import os.path
import shutil

import fs.path

from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath
from pdart.pds4.HstFilename import HstFilename


def _bundle_dir(bundle_id, archive_dir):
    bundle_name = 'hst_%05d' % bundle_id
    return fs.path.join(archive_dir, bundle_name)


def create_bundle_dir(bundle_id, archive_dir):
    # type: (int, unicode) -> None
    bundle_dir = _bundle_dir(bundle_id, archive_dir)
    if os.path.isdir(bundle_dir):
        # handle "it already exists" case
        pass
    elif os.path.isfile(bundle_dir):
        raise Exception('intended base directory %s exists and is a file' %
                        bundle_dir)
    else:
        os.mkdir(bundle_dir)


def create_bundle_db(bundle_id, archive_dir):
    # type: (unicode) -> BundleDB
    bundle_dir = _bundle_dir(bundle_id, archive_dir)
    db = create_bundle_db_from_os_filepath(fs.path.join(bundle_dir,
                                                        _BUNDLE_DB_NAME))
    db.create_tables()
    return db


def copy_downloaded_files(bundle_id, download_root, archive_dir):
    # type: (int, unicode, unicode) -> None

    def is_dot_file(filename):
        # type: (unicode) -> bool
        return filename[0] == '.'

    for (dirpath, dirnames, filenames) in os.walk(download_root):
        filenames = [filename for filename in filenames if
                     not is_dot_file(filename)]
        if len(filenames) > 0:
            rel_dirpath = fs.path.relativefrom(download_root, dirpath)
            path = fs.path.iteratepath(rel_dirpath)
            depth = len(path)
            assert depth == 4, path
            hst, _, _, hst_name = path
            hst_name = hst_name.lower()
            for filename in filenames:
                _, ext = fs.path.splitext(filename)
                assert ext == '.fits'
                hst_filename = HstFilename(filename)

                coll = 'data_%s_%s' % (hst_filename.instrument_name(),
                                       hst_filename.suffix())

                old_path = fs.path.join(dirpath, filename)
                new_dirpath = fs.path.join(archive_dir, hst, coll,
                                           hst_name, 'v$1')
                new_path = fs.path.join(new_dirpath, filename)
                os.makedirs(new_dirpath)
                shutil.copy(old_path, new_path)

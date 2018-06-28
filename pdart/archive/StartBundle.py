import os
import os.path
import shutil

import fs.path
from typing import TYPE_CHECKING

from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from typing import List
    from pdart.new_db.BundleDB import BundleDB

_INITIAL_VID = VID('1.0')  # type: VID
_INITIAL_VID_DIR = 'v$1.0'  # type: unicode


def _create_lidvid_from_parts(parts):
    # type: (List[str]) -> str
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def _bundle_dir(bundle_id, archive_dir):
    # type: (int, unicode) -> unicode
    bundle_name = 'hst_%05d' % bundle_id
    return fs.path.join(archive_dir, bundle_name)


def _bundle_lid(bundle_id):
    # type: (int) -> LID
    bundle_name = 'hst_%05d' % bundle_id
    return LID.create_from_parts([bundle_name])


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
    # type: (int, unicode) -> BundleDB
    bundle_dir = _bundle_dir(bundle_id, archive_dir)
    db = create_bundle_db_from_os_filepath(fs.path.join(bundle_dir,
                                                        _BUNDLE_DB_NAME))
    db.create_tables()

    bundle_name = 'hst_%05d' % bundle_id
    bundle_lidvid = _create_lidvid_from_parts([bundle_name])
    db.create_bundle(bundle_lidvid)
    return db


def copy_downloaded_files(bundle_db, bundle_id, download_root, archive_dir):
    # type: (BundleDB, int, unicode, unicode) -> None

    def is_dot_file(filename):
        # type: (unicode) -> bool
        return filename[0] == '.'

    for (dirpath, dirnames, filenames) in os.walk(download_root):
        # ignore .DB_Store files used by the Mac OS Finder
        filenames = [filename for filename in filenames if
                     not is_dot_file(filename)]
        if len(filenames) > 0:
            rel_dirpath = fs.path.relativefrom(download_root, dirpath)
            path = fs.path.iteratepath(rel_dirpath)
            depth = len(path)
            assert depth == 4, path
            bundle, _, _, hst_name = path
            bundle_lidvid = _create_lidvid_from_parts([bundle])

            product = hst_name.lower()
            for filename in filenames:
                _, ext = fs.path.splitext(filename)
                assert ext == '.fits'
                hst_filename = HstFilename(filename)

                collection = 'data_%s_%s' % (hst_filename.instrument_name(),
                                             hst_filename.suffix())

                old_path = fs.path.join(dirpath, filename)
                new_dirpath = fs.path.join(archive_dir, bundle, collection,
                                           product, _INITIAL_VID_DIR)
                new_path = fs.path.join(new_dirpath, filename)
                os.makedirs(new_dirpath)
                shutil.copy(old_path, new_path)

                # create the collection database object if necessary
                collection_lidvid = _create_lidvid_from_parts([bundle,
                                                               collection])
                bundle_db.create_non_document_collection(collection_lidvid,
                                                         bundle_lidvid)

                # create the product database object
                product_lidvid = _create_lidvid_from_parts(
                    [bundle, collection, product]
                )
                bundle_db.create_fits_product(product_lidvid,
                                              collection_lidvid)

                populate_database_from_fits_file(bundle_db, new_path,
                                                 product_lidvid)

import re

import fs.copy
import fs.path
from fs.osfs import OSFS
from typing import TYPE_CHECKING

from pdart.fs.V1FS import V1FS
from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB

_INITIAL_VID = VID('1.0')  # type: VID


def bundle_to_int(bundle_id):
    # type: (unicode) -> int
    """Converts a bundle_id to an int"""
    res = re.match(r'^hst_([0-9]{5})$', bundle_id)
    if res:
        return int(res.group(1))
    else:
        return None


def copy_files_from_download(src_dir, archive_dir):
    # type: (unicode, unicode) -> int
    """
    Copies the files from a MAST download directory to a new bundle
    directory.  The new archive uses a multi-version hierarchy to
    store the files.
    """
    src_fs = OSFS(src_dir)
    archive_fs = V1FS(archive_dir)
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
        archive_fs.makedirs(dirs)
        fs.copy.copy_file(src_fs, filepath, archive_fs, new_path)
    if False:
        print 'Archive, seen as uni-versioned, looks like this:'
        archive_fs.tree()
        print 'Archive, seen as multi-versioned, looks like this:'
        OSFS(archive_dir).tree()
    return bundle_to_int(bundle)


def _create_lidvid_from_parts(parts):
    # type: (List[str]) -> str
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def _bundle_dir(bundle_id, archive_dir):
    # type: (int, unicode) -> unicode
    bundle_name = 'hst_%05d' % bundle_id
    return fs.path.join(archive_dir, bundle_name)


def create_bundle_db(bundle_id, archive_dir):
    # type: (int, unicode) -> BundleDB
    """
    Creates a new, empty BundleDB in the bundle directory.  The
    database is written to hst_<bundle_id>/<bundle_db_name>.
    """
    bundle_dir = _bundle_dir(bundle_id, archive_dir)
    db = create_bundle_db_from_os_filepath(fs.path.join(bundle_dir,
                                                        _BUNDLE_DB_NAME))
    db.create_tables()

    bundle_name = 'hst_%05d' % bundle_id
    bundle_lidvid = _create_lidvid_from_parts([bundle_name])
    db.create_bundle(bundle_lidvid)
    return db


def populate_database(bundle_id, bundle_db, archive_dir):
    # type: (int, BundleDB, unicode) -> None
    """
    Populates the database with collections, products, and the FITS
    files' contents using the contents of the bundle directory in the
    given archive.
    """
    bundle = 'hst_%05d' % bundle_id
    bundle_lidvid = _create_lidvid_from_parts([bundle])
    v1fs = V1FS(archive_dir)
    for collection in v1fs.listdir(bundle):
        if '$' in collection:
            continue
        collection_path = fs.path.join(bundle, collection)
        collection_lidvid = _create_lidvid_from_parts([bundle, collection])
        bundle_db.create_non_document_collection(collection_lidvid,
                                                 bundle_lidvid)
        for product in v1fs.listdir(collection_path):
            if '$' in product:
                continue
            product_path = fs.path.join(collection_path, product)
            product_lidvid = _create_lidvid_from_parts(
                [bundle, collection, product])
            bundle_db.create_fits_product(product_lidvid, collection_lidvid)
            for fits_file in v1fs.listdir(product_path):
                if '$' in fits_file:
                    continue
                assert fs.path.splitext(fits_file)[1] == '.fits'
                fits_file_path = fs.path.join(product_path, fits_file)
                bundle_db.create_fits_product(product_lidvid,
                                              collection_lidvid)
                fits_sys_path = v1fs.getsyspath(fits_file_path)
                populate_database_from_fits_file(bundle_db, fits_sys_path,
                                                 product_lidvid)


def start_bundle(src_dir, archive_dir):
    # type: (unicode, unicode) -> None
    """
    Create the first version of a bundle from a MAST download.
    """
    bundle_id = copy_files_from_download(src_dir, archive_dir)
    bundle_db = create_bundle_db(bundle_id, archive_dir)
    populate_database(bundle_id, bundle_db, archive_dir)
    # TODO: more to do

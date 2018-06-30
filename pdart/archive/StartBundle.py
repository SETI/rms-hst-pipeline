import os
import os.path
import shutil

import fs.osfs
import fs.path
import fs.tempfs
import pdart.add_pds_tools
import picmaker  # need to precede this with 'import pdart.add_pds_tools'
from typing import TYPE_CHECKING

from pdart.fs.CopyOnWriteFS import CopyOnWriteFS
from pdart.fs.DirUtils import lidvid_to_dir
from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView
from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES

if TYPE_CHECKING:
    from typing import List, Tuple
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


def create_bundle_version_view_and_db(bundle_id, archive_dir):
    # type: (int, unicode) -> Tuple[CopyOnWriteFS, BundleDB]
    '''Return a read/write view of a bundle version.'''
    empty_fs = fs.tempfs.TempFS()
    osfs = fs.osfs.OSFS(archive_dir)
    cow = CopyOnWriteFS(empty_fs, osfs)
    multiversion_fs = MultiversionBundleFS(cow)
    bundle_name = 'hst_%05d' % bundle_id
    bundle_lidvid = LIDVID(_create_lidvid_from_parts([bundle_name]))
    multiversion_fs.make_lidvid_directories(bundle_lidvid)
    view = VersionView(bundle_lidvid, multiversion_fs)
    bundle_db = create_bundle_db(bundle_id, archive_dir)
    return (view, bundle_db)


def create_bundle_dir(bundle_id, archive_dir):
    # type: (int, unicode) -> unicode
    bundle_dir = _bundle_dir(bundle_id, archive_dir)
    if os.path.isdir(bundle_dir):
        # handle "it already exists" case
        pass
    elif os.path.isfile(bundle_dir):
        raise Exception('intended base directory %s exists and is a file' %
                        bundle_dir)
    else:
        os.mkdir(bundle_dir)
    return bundle_dir


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


def make_browse_collections(bundle_db, bundle_id, archive_dir):
    # type: (BundleDB, int, unicode) -> None
    bundle_lidvid = str(bundle_db.get_bundle().lidvid)

    # Get all the collections at once at the start, since we're adding
    # to them.
    for collection in list(bundle_db.get_bundle_collections(bundle_lidvid)):
        if collection.suffix in RAW_SUFFIXES:
            # we need a browse collection
            raw_collection_lidvid = LIDVID(collection.lidvid)
            browse_collection_lidvid = LIDVID.create_from_lid_and_vid(
                raw_collection_lidvid.lid().to_browse_lid(),
                raw_collection_lidvid.vid())
            bundle_db.create_non_document_collection(
                str(browse_collection_lidvid),
                bundle_lidvid)
            for fits_product in bundle_db.get_collection_products(
                    collection.lidvid):
                fits_product_lidvid = LIDVID(fits_product.lidvid)
                browse_product_lidvid = LIDVID.create_from_lid_and_vid(
                    fits_product_lidvid.lid().to_browse_lid(),
                    fits_product_lidvid.vid())
                bundle_db.create_browse_product(str(browse_product_lidvid),
                                                fits_product.lidvid,
                                                str(browse_collection_lidvid))

                for file in bundle_db.get_product_files(fits_product.lidvid):
                    # create browse file in the filesystem
                    file_basename = file.basename
                    browse_basename = os.path.splitext(file_basename)[
                                          0] + '.jpg'

                    fits_filepath = fs.path.join(
                        archive_dir +
                        lidvid_to_dir(fits_product_lidvid),
                        file_basename)
                    browse_product_dir = fs.path.join(
                        archive_dir +
                        lidvid_to_dir(browse_product_lidvid))
                    os.makedirs(browse_product_dir)
                    picmaker.ImagesToPics([str(fits_filepath)],
                                          str(browse_product_dir),
                                          filter="None",
                                          percentiles=(1, 99))
                    browse_filepath = fs.path.join(browse_product_dir,
                                                   browse_basename)
                    # create browse file record in the database
                    size = os.stat(browse_filepath).st_size
                    bundle_db.create_browse_file(browse_basename,
                                                 str(browse_product_lidvid),
                                                 size)


def make_document_collection(bundle_db, bundle_id):
    # type: (BundleDB, int) -> None
    # TODO mucho to do
    pass

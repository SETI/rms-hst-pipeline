import os
import shutil
import tempfile
from typing import TYPE_CHECKING
import unittest

from fs.osfs import OSFS
import fs.path

from pdart.archive.StartBundle import _INITIAL_VID, \
    _create_lidvid_from_parts, bundle_to_int, copy_files_from_download, \
    create_browse_products, create_bundle_db, create_document_collection, \
    create_pds4_labels, populate_database
from pdart.fs.DirUtils import lid_to_dir
from pdart.fs.V1FS import V1FS, _V1_0
from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.new_db.BundleDB import _BUNDLE_DB_NAME
from pdart.new_labels.CollectionLabel import get_collection_inventory_name, \
    get_collection_label_name

if TYPE_CHECKING:
    from sqlalchemy.schema import Column


_DOC_FILES = {u'phase2.pro', u'phase2.pdf', u'phase2.apt'}
# type: Set[unicode]


def _path_to_testfiles():
    # type: () -> unicode
    """Return the path to files needed for testing."""
    return os.path.join(os.path.dirname(__file__), 'testfiles')


def _list_rel_filepaths(root_dir):
    # type: (unicode) -> List[unicode]
    def _list_rel_filepaths_gen():
        for (dirpath, _, filenames) in os.walk(root_dir):
            rel_dirpath = fs.path.relativefrom(root_dir, dirpath)
            for filename in filenames:
                yield os.path.join(rel_dirpath, filename)

    return sorted(_list_rel_filepaths_gen())


class TestStartBundle(unittest.TestCase):
    def setUp(self):
        self.archive_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.archive_dir)

    def test_bundle_to_int(self):
        self.assertEqual(bundle_to_int('hst_01234'), 1234)
        self.assertFalse(bundle_to_int('george'))
        self.assertFalse(bundle_to_int('hst_0000'))
        self.assertFalse(bundle_to_int('hst_000000'))

    def test_copy_files_from_download(self):
        # type: () -> None
        download_dir = fs.path.join(_path_to_testfiles(),
                                    'download_dir')

        res = copy_files_from_download(download_dir, 'hst_13012',
                                       self.archive_dir)
        # check that it read the expected bundle
        self.assertEqual(res, 13012)

        # Check that the newly created archive directory is in
        # multi-version format and that the bundle has its own
        # directory within the archive directory.
        osfs = OSFS(self.archive_dir)
        BUNDLE_DIR = u'/hst_13012'
        self.assertTrue(osfs.exists(BUNDLE_DIR))

        BUNDLE_VERSION_1_DIR = fs.path.join(BUNDLE_DIR, _V1_0)
        self.assertTrue(osfs.exists(BUNDLE_VERSION_1_DIR))

        SUBDIR_VERSIONS = fs.path.join(BUNDLE_DIR, _V1_0,
                                       SUBDIR_VERSIONS_FILENAME)
        self.assertTrue(osfs.exists(SUBDIR_VERSIONS))

    def test_create_bundle_db(self):
        # type: () -> None
        download_dir = fs.path.join(_path_to_testfiles(),
                                    'download_dir')
        copy_files_from_download(download_dir, 'hst_13012', self.archive_dir)
        db = create_bundle_db(13012, self.archive_dir)
        try:
            # returns the DB
            self.assertTrue(db)
            db_filename = os.path.join(self.archive_dir,
                                       'hst_13012',
                                       _BUNDLE_DB_NAME)
            # creates the DB file
            self.assertTrue(os.path.isfile(db_filename))

            bundle_lid = LID.create_from_parts(['hst_13012'])
            bundle_lidvid = LIDVID.create_from_lid_and_vid(
                bundle_lid,
                _INITIAL_VID)
            self.assertTrue(db.bundle_exists(str(bundle_lidvid)))
            BUNDLE_LIDVID = 'urn:nasa:pds:hst_13012::1.0'
            self.assertTrue(db.bundle_exists(BUNDLE_LIDVID))
        finally:
            db.close()

    def test_populate_database(self):
        download_dir = fs.path.join(_path_to_testfiles(),
                                    'download_dir')

        copy_files_from_download(download_dir, 'hst_13012', self.archive_dir)
        db = create_bundle_db(13012, self.archive_dir)
        try:
            populate_database(13012, db, self.archive_dir)
            # Test it.
            archive_fs = V1FS(self.archive_dir)
            for coll_dir in archive_fs.listdir('/hst_13012'):
                if '$' in coll_dir:
                    continue
                coll_lidvid = 'urn:nasa:pds:hst_13012:%s::1.0' % coll_dir
                self.assertTrue(db.collection_exists(coll_lidvid))
            # I'm going to leave the testing at this high level and
            # assume if it's correct as far down as the collections,
            # the rest is too.
        finally:
            db.close()

    def test_create_browse_products(self):
        download_dir = fs.path.join(_path_to_testfiles(),
                                    'download_dir')

        copy_files_from_download(download_dir, 'hst_13012', self.archive_dir)
        db = create_bundle_db(13012, self.archive_dir)
        try:
            populate_database(13012, db, self.archive_dir)
            create_browse_products(13012, db, self.archive_dir)

            # Check that the browse collection, product, and file
            # exist in the database.
            parts = ['hst_13012', 'browse_acs_flt', 'jbz504ejq']
            browse_file_basename = u'jbz504ejq_flt.jpg'
            expected_collection_lidvid = _create_lidvid_from_parts(parts[:2])
            self.assertTrue(db.non_document_collection_exists(
                    expected_collection_lidvid))
            expected_product_lidvid = _create_lidvid_from_parts(parts)
            self.assertTrue(db.browse_product_exists(expected_product_lidvid))
            self.assertTrue(db.browse_file_exists(browse_file_basename,
                                                  expected_product_lidvid))

            # Check that they also exist in the filesystem.

            # We don't need to check the intermediate directories; we
            # can test for the browse file and verify them all in one
            # fell swoop.
            browse_file_fs_path = fs.path.join('/'.join(parts),
                                               browse_file_basename)
            self.assertTrue(V1FS(self.archive_dir).isfile(browse_file_fs_path))
        finally:
            db.close()

    def test_create_document_collection(self):
        # type: () -> None
        download_dir = fs.path.join(_path_to_testfiles(),
                                    'download_dir')
        documents_dir = fs.path.join(_path_to_testfiles(), 'document_files')

        copy_files_from_download(download_dir, 'hst_13012', self.archive_dir)
        db = create_bundle_db(13012, self.archive_dir)
        try:
            archive_fs = V1FS(self.archive_dir)
            populate_database(13012, db, self.archive_dir)

            create_document_collection(
                13012, db, self.archive_dir,
                documents_dir,
                _DOC_FILES)

            # ensure the collection exists in the database
            document_collection_lidvid = _create_lidvid_from_parts(
                ['hst_13012', 'document'])
            self.assertTrue(db.document_collection_exists(
                    document_collection_lidvid))

            # ensure the collection exists in the filesystem
            document_collection_dir = u'/hst_13012/document'
            self.assertTrue(archive_fs.isdir(
                    document_collection_dir))

            # ensure the product exists in the database
            document_product_lidvid = _create_lidvid_from_parts(
                ['hst_13012', 'document', 'phase2'])
            self.assertTrue(db.document_product_exists(
                    document_product_lidvid))

            # ensure the product exists in the filesystem
            document_product_dir = u'/hst_13012/document/phase2'
            self.assertTrue(archive_fs.isdir(
                    document_product_dir))

            # ensure the files exist in the database
            for basename in _DOC_FILES:
                self.assertTrue(db.document_file_exists(
                        basename,
                        document_product_lidvid))

            # ensure the files exist in the filesystem
            for basename in _DOC_FILES:
                document_file_filepath = fs.path.join(document_product_dir,
                                                      basename)
                self.assertTrue(archive_fs.isfile(document_file_filepath))
        finally:
            db.close()

    def test_create_pds4_labels(self):
        # type: () -> None
        download_dir = fs.path.join(_path_to_testfiles(),
                                    'download_dir')
        documents_dir = fs.path.join(_path_to_testfiles(), 'document_files')

        copy_files_from_download(download_dir, 'hst_13012', self.archive_dir)
        db = create_bundle_db(13012, self.archive_dir)
        try:
            populate_database(13012, db, self.archive_dir)
            create_browse_products(13012, db, self.archive_dir)

            create_document_collection(
                13012, db, self.archive_dir,
                documents_dir,
                _DOC_FILES)

            create_pds4_labels(13012, db, self.archive_dir)

            # Test that all the labels exist
            vfs = V1FS(self.archive_dir)

            def lidvid_to_dir(lidvid):
                # type: (str) -> unicode
                def get_lid(lidvid):
                    # type: (str) -> LID
                    return LIDVID(lidvid).lid()

                return lid_to_dir(get_lid(str(lidvid)))

            # check that product labels exist
            bundle = db.get_bundle()
            bundle_lidvid = str(bundle.lidvid)
            bundle_dir = lidvid_to_dir(bundle_lidvid)
            for collection in db.get_bundle_collections(bundle_lidvid):
                collection_lidvid = str(collection.lidvid)
                for product in db.get_collection_products(collection_lidvid):
                    product_lidvid = str(product.lidvid)
                    product_dir = lidvid_to_dir(product_lidvid)
                    if db.browse_product_exists(product_lidvid):
                        # handle browse product
                        files = list(db.get_product_files(product_lidvid))
                        self.assertEquals(1, len(files))
                        file = files[0]
                        label_base = fs.path.splitext(file.basename)[0]
                        label_filename = label_base + u'.xml'
                        label_filepath = fs.path.join(product_dir,
                                                      label_filename)
                        self.assertTrue(vfs.isfile(label_filepath))
                    elif db.document_product_exists(product_lidvid):
                        # handle document product
                        label_filename = 'phase2.xml'
                        label_filepath = fs.path.join(product_dir,
                                                      label_filename)
                        vfs.tree()
                        self.assertTrue(vfs.isfile(label_filepath),
                                        label_filepath)
                    elif db.fits_product_exists(product_lidvid):
                        # handle fits product
                        files = list(db.get_product_files(product_lidvid))
                        self.assertEquals(1, len(files))
                        file = files[0]
                        label_base = fs.path.splitext(file.basename)[0]
                        label_filename = label_base + u'.xml'
                        label_filepath = fs.path.join(product_dir,
                                                      label_filename)
                        self.assertTrue(vfs.isfile(label_filepath))
                    else:
                        assert False, ('missing case: ' + product_lidvid)

                collection_dir = lidvid_to_dir(collection_lidvid)
                label_filename = get_collection_label_name(db,
                                                           collection_lidvid)
                label_filepath = fs.path.join(collection_dir, label_filename)
                self.assertTrue(vfs.isfile(label_filepath), label_filepath)
                inventory_filename = get_collection_inventory_name(
                    db, collection_lidvid)
                inventory_filepath = fs.path.join(collection_dir,
                                                  inventory_filename)
                self.assertTrue(vfs.isfile(inventory_filepath),
                                inventory_filepath)
            label_filepath = fs.path.join(bundle_dir, u'bundle.xml')
            self.assertTrue(vfs.isfile(label_filepath), label_filepath)
        finally:
            db.close()

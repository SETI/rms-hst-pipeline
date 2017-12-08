import unittest

from pdart.new_db.BundleDB import *
from pdart.new_db.SqlAlchTables import Base

if TYPE_CHECKING:
    from typing import Set

_TABLES = {'bundles',
           'collections', 'document_collections', 'non_document_collections',
           'products', 'browse_products', 'document_products', 'fits_products',
           'files', 'fits_files', 'bad_fits_files', 'document_files',
           'hdus', 'cards'}  # type: Set[str]


class Test_BundleDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_BundleDB(self):
        # type: () -> None
        db = self.db
        self.assertTrue(db.is_open())
        metadata = Base.metadata
        self.assertEqual(set(metadata.tables.keys()), _TABLES)
        db.close()
        self.assertFalse(db.is_open())

    def test_create_bundle(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.assertTrue(self.db.session.query(Bundle).filter(
            Bundle.lidvid == bundle_lidvid).count() == 0)
        self.assertFalse(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.session.query(Bundle).filter(
            Bundle.lidvid == bundle_lidvid).count() == 1)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))

    def test_get_bundle(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)
        bundle = self.db.get_bundle(bundle_lidvid)
        self.assertEqual(bundle_lidvid, bundle.lidvid)
        self.assertEqual(99999, bundle.proposal_id)

    def test_get_collection(self):
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        collection = self.db.get_collection(collection_lidvid)
        self.assertEqual(NonDocumentCollection, type(collection))
        self.assertEqual('acs', collection.instrument)
        self.assertEqual('raw', collection.suffix)

    def test_get_product(self):
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        product = self.db.get_product(product_lidvid)
        self.assertEqual(product_lidvid, product.lidvid)
        self.assertEqual(collection_lidvid, product.collection_lidvid)

    def test_create_non_document_collection(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.assertFalse(
            self.db.non_document_collection_exists(collection_lidvid))

        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        self.assertTrue(
            self.db.non_document_collection_exists(collection_lidvid))

        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)
        self.assertTrue(
            self.db.non_document_collection_exists(collection_lidvid))

    def test_create_fits_product(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.assertFalse(self.db.fits_product_exists(product_lidvid))

        self.db.create_fits_product(product_lidvid, collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

        self.db.create_fits_product(product_lidvid, collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

    def test_create_fits_file(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_99999::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:hst_99999:data_acs_raw:p::8.1'
        self.db.create_fits_product(product_lidvid, collection_lidvid)

        basename = 'file.fits'
        self.assertFalse(self.db.fits_file_exists(basename, product_lidvid))

        self.db.create_fits_file(basename, product_lidvid, 1)
        self.assertTrue(self.db.fits_file_exists(basename, product_lidvid))

        self.db.create_fits_file(basename, product_lidvid, 1)
        self.assertTrue(self.db.fits_file_exists(basename, product_lidvid))

    def test_exploratory(self):
        # type: () -> None
        # what happens if you create tables twice?
        self.db.create_tables()
        # no exception, at least

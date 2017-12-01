import unittest

from pdart.new_db.BundleDB import *
from pdart.new_db.SqlAlchTables import Base

if TYPE_CHECKING:
    from typing import Set

_TABLES = {'bundles',
           'collections', 'document_collections', 'non_document_collections',
           'products', 'browse_products', 'document_products', 'fits_products',
           'document_files', 'hdus', 'cards'}  # type: Set[str]


class Test_BundleDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = BundleDB.create_database_in_memory()
        self.db.create_tables()

    def test_BundleDB(self):
        # type: () -> None
        db = self.db
        self.assertTrue(db.is_open())
        metadata = Base.metadata
        self.assertEqual(set(metadata.tables.keys()), _TABLES)
        db.close()
        self.assertFalse(db.is_open())

    def test_upsert_bundle(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:b::1.1'
        self.assertTrue(self.db.session.query(Bundle).filter(
            Bundle.lidvid == bundle_lidvid).count() == 0)
        self.assertFalse(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.session.query(Bundle).filter(
            Bundle.lidvid == bundle_lidvid).count() == 1)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))
        self.db.create_bundle(bundle_lidvid)
        self.assertTrue(self.db.bundle_exists(bundle_lidvid))

    def test_upsert_non_document_collection(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:b::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:b:c::1.8'
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

    def test_upsert_fits_product(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:b::1.1'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = 'urn:nasa:pds:b:c::1.8'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        product_lidvid = 'urn:nasa:pds:b:c:p::8.1'
        self.assertFalse(self.db.fits_product_exists(product_lidvid))

        self.db.create_fits_product(product_lidvid, collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

        self.db.create_fits_product(product_lidvid, collection_lidvid)
        self.assertTrue(self.db.fits_product_exists(product_lidvid))

    def test_exploratory(self):
        # type: () -> None
        # what happens if you create tables twice?
        self.db.create_tables()
        # no exception, at least

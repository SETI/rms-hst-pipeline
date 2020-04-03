import unittest

from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.labels.CollectionInventory import make_collection_inventory
from pdart.labels.Utils import assert_golden_file_equal

_BUNDLE_LIDVID = "urn:nasa:pds:hst_09059::1.3"
_COLLECTION_LIDVID = "urn:nasa:pds:hst_09059:data_acs_raw::1.4"
_FITS_PRODUCT_LIDVID = "urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.5"


class Test_CollectionInventory(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        self.db.create_non_document_collection(_COLLECTION_LIDVID, _BUNDLE_LIDVID)

        self.db.create_fits_product(_FITS_PRODUCT_LIDVID, _COLLECTION_LIDVID)

    def test_make_collection_inventory(self) -> None:
        inventory = make_collection_inventory(self.db, _COLLECTION_LIDVID)

        assert_golden_file_equal(self, "test_CollectionInventory.golden.txt", inventory)

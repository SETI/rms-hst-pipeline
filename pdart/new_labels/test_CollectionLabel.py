import unittest

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.CollectionLabel import *
from pdart.new_labels.Utils import golden_file_contents

_BUNDLE_LIDVID = 'urn:nasa:pds:hst_09059::1.3'
_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw::1'
_FITS_PRODUCT_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1'


class Test_CollectionLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        self.db.create_non_document_collection(_COLLECTION_LIDVID,
                                               _BUNDLE_LIDVID)

        self.db.create_fits_product(_FITS_PRODUCT_LIDVID, _COLLECTION_LIDVID)

    def test_make_collection_inventory(self):
        # type: () -> None
        inventory = make_collection_inventory(self.db,
                                              _COLLECTION_LIDVID)
        print inventory
        expected_inventory = golden_file_contents(
            'test_CollectionLabel_inventory.golden.txt')
        self.assertEqual(expected_inventory, inventory)

    def test_make_collection_label(self):
        # type: () -> None
        label = make_collection_label(self.db,
                                      _COLLECTION_LIDVID,
                                      True)
        print label
        expected_label = golden_file_contents(
            'test_CollectionLabel.golden.xml')
        self.assertEqual(expected_label, label)

import unittest

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.BundleLabel import *
from pdart.new_labels.Utils import golden_file_contents

_BUNDLE_LIDVID = 'urn:nasa:pds:hst_09059::1.3'
_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw::1.2'
_FITS_PRODUCT_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.2'


class Test_BundleLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        self.db.create_non_document_collection(_COLLECTION_LIDVID,
                                               _BUNDLE_LIDVID)

        self.db.create_fits_product(_FITS_PRODUCT_LIDVID, _COLLECTION_LIDVID)

    def test_make_bundle_label(self):
        # type: () -> None
        label = make_bundle_label(self.db, True)
        expected_label = golden_file_contents('test_BundleLabel.golden.xml')
        self.assertEqual(expected_label, label)

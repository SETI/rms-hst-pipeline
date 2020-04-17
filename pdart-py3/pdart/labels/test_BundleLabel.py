import unittest

from pdart.citations import Citation_Information
from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.labels.BundleLabel import make_bundle_label
from pdart.labels.Utils import assert_golden_file_equal

_BUNDLE_LIDVID = "urn:nasa:pds:hst_09059::1.3"
_COLLECTION_LIDVID = "urn:nasa:pds:hst_09059:data_acs_raw::1.2"
_FITS_PRODUCT_LIDVID = "urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.2"


class Test_BundleLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        # TODO add a document collection to check reference_types
        self.db.create_other_collection(_COLLECTION_LIDVID, _BUNDLE_LIDVID)

        self.db.create_fits_product(_FITS_PRODUCT_LIDVID, _COLLECTION_LIDVID)
        self.info = Citation_Information.create_test_citation_information()

    def test_make_bundle_label(self) -> None:
        label = make_bundle_label(self.db, self.info, True)
        assert_golden_file_equal(self, "test_BundleLabel.golden.xml", label)

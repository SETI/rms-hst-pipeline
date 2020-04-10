import unittest

from pdart.citations import Citation_Information
from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.labels.DocumentProductLabel import make_document_product_label
from pdart.labels.Utils import assert_golden_file_equal


class Test_DocumentProductLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.info = Citation_Information.create_test_citation_information()

    def test_make_document_product_label(self) -> None:
        bundle_lidvid = "urn:nasa:pds:hst_13012::1.0"
        self.db.create_bundle("urn:nasa:pds:hst_13012::1.0")

        collection_lidvid = "urn:nasa:pds:hst_13012:document::3.14159"
        self.db.create_document_collection(collection_lidvid, bundle_lidvid)

        document_product_lidvid = "urn:nasa:pds:hst_13012:document:phase2::1.0"

        label = make_document_product_label(
            self.db, self.info, document_product_lidvid, True, "2017-02-31"
        )

        assert_golden_file_equal(self, "test_DocumentProductLabel.golden.xml", label)

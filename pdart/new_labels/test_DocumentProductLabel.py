import unittest

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.DocumentProductLabel import *
from pdart.new_labels.Utils import golden_file_contents


class Test_DocumentProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_make_document_product_label(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_13012::1.0'
        self.db.create_bundle('urn:nasa:pds:hst_13012::1.0')

        collection_lidvid = \
            'urn:nasa:pds:hst_13012:document::3.14159'
        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)

        document_product_lidvid = 'urn:nasa:pds:hst_13012:document:phase2::1.0'

        label = make_document_product_label(self.db,
                                            document_product_lidvid,
                                            True,
                                            '2017-02-31')

        expected = golden_file_contents('test_DocumentProductLabel.golden.xml')
        self.assertEqual(expected, label)

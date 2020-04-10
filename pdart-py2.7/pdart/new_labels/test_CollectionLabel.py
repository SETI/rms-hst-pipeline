import unittest

from Citation_Information import Citation_Information
from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.CollectionLabel import make_collection_label
from pdart.new_labels.Utils import assert_golden_file_equal

_BUNDLE_LIDVID = 'urn:nasa:pds:hst_09059::1.3'
_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw::1.2'
_FITS_PRODUCT_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.2'

_DOC_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:document::1.2'
_DOC_PRODUCT_LIDVID = 'urn:nasa:pds:hst_09059:document:phase2::1.2'


class Test_CollectionLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        self.db.create_non_document_collection(_COLLECTION_LIDVID,
                                               _BUNDLE_LIDVID)

        self.db.create_fits_product(_FITS_PRODUCT_LIDVID, _COLLECTION_LIDVID)

        self.db.create_document_collection(_DOC_COLLECTION_LIDVID,
                                           _BUNDLE_LIDVID)

        self.db.create_document_product(_DOC_PRODUCT_LIDVID,
                                        _DOC_COLLECTION_LIDVID)
        self.info = Citation_Information.create_test_citation_information()

    def test_make_collection_label(self):
        # type: () -> None

        # make a standard collection label
        label = make_collection_label(self.db, self.info,
                                      _COLLECTION_LIDVID,
                                      True)
        assert_golden_file_equal(self,
                                 'test_CollectionLabel.golden.xml',
                                 label)

    def test_make_doc_collection_label(self):
        # type: () -> None

        # make a documentation collection label
        label = make_collection_label(self.db, self.info,
                                      _DOC_COLLECTION_LIDVID,
                                      True)

        assert_golden_file_equal(self,
                                 'test_DocCollectionLabel.golden.xml',
                                 label)

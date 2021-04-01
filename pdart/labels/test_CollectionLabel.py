import unittest

from typing import List, Tuple

from pdart.citations import Citation_Information
from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.labels.CollectionLabel import make_collection_label
from pdart.labels.Utils import assert_golden_file_equal

_BUNDLE_LIDVID = "urn:nasa:pds:hst_09059::1.3"
_COLLECTION_LIDVID = "urn:nasa:pds:hst_09059:data_acs_raw::1.2"
_FITS_PRODUCT_LIDVID = "urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::1.2"

_DOC_COLLECTION_LIDVID = "urn:nasa:pds:hst_09059:document::1.2"
_DOC_PRODUCT_LIDVID = "urn:nasa:pds:hst_09059:document:phase2::1.2"


class Test_CollectionLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        self.db.create_other_collection(_COLLECTION_LIDVID, _BUNDLE_LIDVID)

        self.db.create_fits_product(_FITS_PRODUCT_LIDVID, _COLLECTION_LIDVID)

        self.db.create_document_collection(_DOC_COLLECTION_LIDVID, _BUNDLE_LIDVID)

        self.db.create_document_product(_DOC_PRODUCT_LIDVID, _DOC_COLLECTION_LIDVID)
        self.info = Citation_Information.create_test_citation_information()

        # Create start/stop time in db for testing purpose
        self.db.update_fits_product_time(
            _FITS_PRODUCT_LIDVID,
            "2005-01-19T14:58:56Z",
            "2005-01-19T15:41:05Z",
        )

        # Create wavelength range in db for testing purpose
        self.db.update_wavelength_range(
            _FITS_PRODUCT_LIDVID,
            ["Visible", "Near Infrared", "Infrared"],
        )

        # Create target identifications db for testing purpose
        target_id = "09059_1"
        target_identifications: List[Tuple] = [
            (
                "762 Pulcova",
                [
                    "(762) Pulcova",
                    "(762) 1913SQ",
                    "1913SQ",
                    "(762) 1952 QM1",
                    "1952 QM1",
                    "Pulcova",
                    "Minor Planet 762",
                    "NAIF ID 2000762",
                ],
                "Asteroid",
                [],
                "urn:nasa:pds:context:target:asteroid.762_pulcova",
            )
        ]
        self.db.add_record_to_target_identification_db(
            target_id, target_identifications
        )

        # Create data collection title in db for testing purpose
        data_collection_title = (
            "Raw, uncalibrated ACS/HRC image files from HST Program 9059."
        )
        self.db.update_fits_product_collection_title(
            _COLLECTION_LIDVID, data_collection_title
        )

    def test_make_collection_label(self) -> None:
        # make a standard collection label
        label = make_collection_label(
            self.db, self.info, _COLLECTION_LIDVID, _BUNDLE_LIDVID, True, True
        )
        print(label)
        assert_golden_file_equal(self, "test_CollectionLabel.golden.xml", label)

    def test_make_doc_collection_label(self) -> None:
        # make a documentation collection label
        label = make_collection_label(
            self.db, self.info, _DOC_COLLECTION_LIDVID, _BUNDLE_LIDVID, True, True
        )

        assert_golden_file_equal(self, "test_DocCollectionLabel.golden.xml", label)

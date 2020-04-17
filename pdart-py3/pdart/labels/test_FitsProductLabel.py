import unittest

from fs.path import basename

from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.labels.FitsProductLabel import make_fits_product_label
from pdart.labels.Utils import assert_golden_file_equal, path_to_testfile


class Test_FitsProductLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_make_fits_product_label(self) -> None:
        bundle_lidvid = "urn:nasa:pds:hst_13012::123.90201"
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw::3.14159"
        self.db.create_other_collection(collection_lidvid, bundle_lidvid)

        fits_product_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.13"
        self.db.create_fits_product(fits_product_lidvid, collection_lidvid)

        os_filepath = path_to_testfile("jbz504eoq_raw.fits")

        populate_database_from_fits_file(self.db, os_filepath, fits_product_lidvid)

        file_basename = basename(os_filepath)

        label = make_fits_product_label(
            self.db, fits_product_lidvid, file_basename, True
        )

        assert_golden_file_equal(self, "test_FitsProductLabel.golden.xml", label)

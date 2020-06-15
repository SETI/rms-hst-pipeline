import os
import tempfile
import unittest

from pdart.db.BrowseFileDB import populate_database_from_browse_file
from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.labels.BrowseProductLabel import make_browse_product_label
from pdart.labels.Utils import assert_golden_file_equal


class Test_BrowseProductLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        (handle, filepath) = tempfile.mkstemp()
        os.close(handle)
        self.dummy_os_filepath = filepath

    def tearDown(self) -> None:
        os.remove(self.dummy_os_filepath)

    def test_make_browse_product_label(self) -> None:
        self.db.create_tables()

        bundle_lidvid = "urn:nasa:pds:hst_13012::123.456"
        self.db.create_bundle(bundle_lidvid)

        fits_collection_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw::3.14159"
        self.db.create_other_collection(fits_collection_lidvid, bundle_lidvid)

        browse_collection_lidvid = "urn:nasa:pds:hst_13012:browse_acs_raw::3.14159"
        self.db.create_other_collection(browse_collection_lidvid, bundle_lidvid)

        fits_product_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.1"
        self.db.create_fits_product(fits_product_lidvid, fits_collection_lidvid)

        browse_product_lidvid = (
            "urn:nasa:pds:hst_13012:browse_acs_raw:jbz504eoq_raw::2.1"
        )
        self.db.create_browse_product(
            browse_product_lidvid, fits_product_lidvid, browse_collection_lidvid
        )

        browse_file_basename = "jbz504eoq_raw.jpg"

        populate_database_from_browse_file(
            self.db,
            browse_product_lidvid,
            fits_product_lidvid,
            browse_collection_lidvid,
            self.dummy_os_filepath,
            browse_file_basename,
            5492356,
        )

        browse_file = self.db.get_file(browse_file_basename, browse_product_lidvid)

        label = make_browse_product_label(
            self.db, browse_product_lidvid, browse_file_basename, True
        )

        assert_golden_file_equal(self, "test_BrowseProductLabel.golden.xml", label)

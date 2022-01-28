import unittest
from typing import cast

from pdart.db.BrowseFileDB import populate_database_from_browse_file
from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.db.SqlAlchTables import BrowseFile
from pdart.db.utils import path_to_testfile


class Test_BrowseFileDB(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_populate_database_from_browse_file(self) -> None:
        fits_product_lidvid = "urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::2.0"
        fits_collection_lidvid = "urn:nasa:pds:hst_09059:data_acs_raw::2.0"
        os_filepath = path_to_testfile("j6gp01lzq_raw.fits")

        self.db.create_fits_product(fits_product_lidvid, fits_collection_lidvid)

        browse_product_lidvid = (
            "urn:nasa:pds:hst_09059:browse_acs_raw:j6gp01lzq_raw::2.0"
        )
        browse_collection_lidvid = "urn:nasa:pds:hst_09059:browse_acs_raw::1.6"
        browse_basename = "j6gp01lzq_raw.jpg"
        byte_size = 12347
        populate_database_from_browse_file(
            self.db,
            browse_product_lidvid,
            fits_product_lidvid,
            browse_collection_lidvid,
            os_filepath,
            browse_basename,
            byte_size,
        )
        self.assertTrue(
            self.db.browse_file_exists(browse_basename, browse_product_lidvid)
        )

        file = self.db.get_file(browse_basename, browse_product_lidvid)
        self.assertTrue(file)
        # lidvid and basename are defined to be right (in get_file())
        self.assertTrue(isinstance(file, BrowseFile))
        browse_file = cast(BrowseFile, file)
        self.assertEqual(byte_size, browse_file.byte_size)

import unittest

from fs.path import join

from pdart.new_db.BrowseFileDB import *
from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_db.SqlAlchTables import BrowseFile


class Test_BrowseFileDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_populate_database_from_browse_file(self):
        # type: () -> None
        archive = '/Users/spaceman/Desktop/Archive'

        fits_product_lidvid = \
            'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::2'
        os_filepath = join(
            archive,
            'hst_09059/data_acs_raw/visit_01/j6gp01lzq_raw.fits')

        populate_database_from_fits_file(self.db,
                                         os_filepath,
                                         fits_product_lidvid)

        browse_product_lidvid = \
            'urn:nasa:pds:hst_09059:browse_acs_raw:j6gp01lzq_raw::2'
        collection_lidvid = \
            'urn:nasa:pds:hst_09059:browse_acs_raw::1.6'
        browse_basename = 'j6gp01lzq_raw.jpg'
        byte_size = 12347
        populate_database_from_browse_file(self.db,
                                           browse_product_lidvid,
                                           collection_lidvid,
                                           browse_basename,
                                           byte_size)
        self.assertTrue(self.db.browse_file_exists(browse_basename,
                                                   browse_product_lidvid))

        file = self.db.get_file(browse_product_lidvid,
                                browse_basename)
        self.assertTrue(file)
        # lidvid and basename are defined to be right (in get_file())
        self.assertTrue(isinstance(file, BrowseFile))
        self.assertEquals(byte_size, file.byte_size)

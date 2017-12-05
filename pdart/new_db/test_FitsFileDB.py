import unittest

from fs.path import join

from pdart.new_db.BundleDB import BundleDB
from pdart.new_db.FitsFileDB import *


class Test_FitsFileDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = BundleDB.create_database_in_memory()
        self.db.create_tables()

    def test_populate_from_fits_file(self):
        # type: () -> None
        archive = '/Users/spaceman/Desktop/Archive'

        fits_product_lidvid = \
            'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::2'
        os_filepath = join(
            archive,
            'hst_09059/data_acs_raw/visit_01/j6gp01lzq_raw.fits')

        populate_from_fits_file(self.db,
                                os_filepath,
                                fits_product_lidvid)
        self.assertTrue(self.db.fits_file_exists(basename(os_filepath),
                                                 fits_product_lidvid))

        self.assertFalse(self.db.bad_fits_file_exists(basename(os_filepath),
                                                      fits_product_lidvid))

        if False:
            self.assertTrue(self.db.hdu_exists(0,
                                               basename(os_filepath),
                                               fits_product_lidvid))

    def test_populate_from_bad_fits_file(self):
        # type: () -> None

        fits_product_lidvid = \
            'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::2'
        os_filepath = 'j6gp01lzq_raw.fits'

        populate_from_fits_file(self.db,
                                os_filepath,
                                fits_product_lidvid)

        self.assertFalse(self.db.fits_file_exists(basename(os_filepath),
                                                  fits_product_lidvid))

        self.assertTrue(self.db.bad_fits_file_exists(basename(os_filepath),
                                                     fits_product_lidvid))

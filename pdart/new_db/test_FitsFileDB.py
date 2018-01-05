import unittest

from fs.path import join

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import *


class Test_FitsFileDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_populate_from_fits_file(self):
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

        file_basename = basename(os_filepath)

        self.assertTrue(self.db.fits_file_exists(file_basename,
                                                 fits_product_lidvid))

        self.assertFalse(self.db.bad_fits_file_exists(file_basename,
                                                      fits_product_lidvid))

        # test that we got some HDUS
        self.assertTrue(self.db.hdu_exists(0,
                                           file_basename,
                                           fits_product_lidvid))

        # test that we got some cards
        self.assertTrue(self.db.card_exists('BITPIX',
                                            0,
                                            fits_product_lidvid))

    def test_populate_from_bad_fits_file(self):
        # type: () -> None

        fits_product_lidvid = \
            'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw::2'
        os_filepath = 'j6gp01lzq_raw.fits'

        populate_database_from_fits_file(self.db,
                                         os_filepath,
                                         fits_product_lidvid)

        self.assertFalse(self.db.fits_file_exists(basename(os_filepath),
                                                  fits_product_lidvid))

        self.assertTrue(self.db.bad_fits_file_exists(basename(os_filepath),
                                                     fits_product_lidvid))

    def test_get_card_dictionaries(self):
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

        file_basename = basename(os_filepath)

        card_dicts = get_card_dictionaries(self.db,
                                           fits_product_lidvid,
                                           file_basename)

        self.assertTrue(card_dicts)
        self.assertEquals(4, len(card_dicts))
        self.assertEquals(16, int(card_dicts[0]['BITPIX']))

    def test_file_offsets(self):
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

        offsets = get_file_offsets(self.db, fits_product_lidvid)
        self.assertEqual(4, len(offsets))

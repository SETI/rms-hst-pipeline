import unittest

from fs.osfs import OSFS
from fs.path import dirname
from fs.tempfs import TempFS

from pdart.browse.CreateBrowse import *
from pdart.fs.CopyOnWriteFS import CopyOnWriteFS
from pdart.new_db.BundleDB import create_bundle_db_in_memory

_FITS_PRODUCT_LID = 'urn:nasa:pds:hst_09059:data_acs_raw:j6gp01lzq_raw'
_FITS_PRODUCT_LIDVID = _FITS_PRODUCT_LID + '::1.6'
_FITS_FILE_PATH = u'/hst_09059/data_acs_raw/j6gp01lzq_raw' \
                  u'/j6gp01lzq_raw.fits'
_FITS_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw::6.12'
_BROWSE_PRODUCT_LID = 'urn:nasa:pds:hst_09059:browse_acs_raw:j6gp01lzq_raw'
_BROWSE_PRODUCT_LIDVID = _BROWSE_PRODUCT_LID + '::2.7'
_BROWSE_FILE_PATH = u'/hst_09059/browse_acs_raw/j6gp01lzq_raw' \
                    u'/j6gp01lzq_raw.jpg'
_BROWSE_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:browse_acs_raw::3.56'


class Test_CreateBrowse(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.there = join(dirname(__file__),
                          '..',
                          'testfiles',
                          'single-version-bundle')

        self.base_fs = OSFS(self.there)
        self.delta_fs = TempFS()
        self.fs = CopyOnWriteFS(self.base_fs, self.delta_fs)

        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_populate_database_from_browse_product(self):
        # type: () -> None
        self.db.create_fits_product(_FITS_PRODUCT_LIDVID,
                                    _FITS_COLLECTION_LIDVID)

        populate_database_from_browse_product(self.db, _BROWSE_FILE_PATH,
                                              234567, _BROWSE_PRODUCT_LIDVID,
                                              _FITS_PRODUCT_LIDVID,
                                              _BROWSE_COLLECTION_LIDVID)
        file_basename = basename(_BROWSE_FILE_PATH)
        self.assertTrue(self.db.browse_file_exists(file_basename,
                                                   _BROWSE_PRODUCT_LIDVID))
        self.assertTrue(self.db.browse_product_exists(_BROWSE_PRODUCT_LIDVID))
        self.assertEqual(234567,
                         self.db.get_file(file_basename,
                                          _BROWSE_PRODUCT_LIDVID).byte_size)

    def test_create_browse_directory(self):
        create_browse_directory(self.fs, _BROWSE_PRODUCT_LID)
        self.assertTrue(self.fs.exists(u'/hst_09059/browse_acs_raw'))
        self.assertTrue(self.fs.isdir(u'/hst_09059/browse_acs_raw'))
        self.assertTrue(self.fs.exists(
            u'/hst_09059/browse_acs_raw/j6gp01lzq_raw'))
        self.assertTrue(self.fs.isdir(
            u'/hst_09059/browse_acs_raw/j6gp01lzq_raw'))

        # check no harm in running it twice
        create_browse_directory(self.fs, _BROWSE_PRODUCT_LID)
        self.assertTrue(self.fs.exists(u'/hst_09059/browse_acs_raw'))
        self.assertTrue(self.fs.isdir(u'/hst_09059/browse_acs_raw'))
        self.assertTrue(self.fs.exists(
            u'/hst_09059/browse_acs_raw/j6gp01lzq_raw'))
        self.assertTrue(self.fs.isdir(
            u'/hst_09059/browse_acs_raw/j6gp01lzq_raw'))

    def test_create_browse_file_from_fits_file(self):
        # type: () -> None
        create_browse_file_from_fits_file(self.fs,
                                          _FITS_PRODUCT_LID,
                                          _BROWSE_PRODUCT_LID)
        self.assertTrue(self.fs.exists(_BROWSE_FILE_PATH))

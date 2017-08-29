from pdart.fs.InitialVersionView import *
import unittest

from fs.memoryfs import MemoryFS

_ROOT = u'/'
# type: unicode
_V1 = u'v$1'
# type: unicode
_BUNDLE_ID = u'hst_00000'
# type: unicode


class TestInitialVersionView(unittest.TestCase):
    def setUp(self):
        self.legacy_fs = MemoryFS()
        self.bundle_id = _BUNDLE_ID
        self.view = InitialVersionView(self.bundle_id, self.legacy_fs)

    def test_bare_fs(self):
        """
        Test that the proper hierarchy is synthesized even for an
        empty legacy filesystem.
        """
        BUNDLE_DIR = join(_ROOT, _BUNDLE_ID)
        self.assertTrue(self.view.exists(_ROOT))
        self.assertTrue(self.view.isdir(_ROOT))
        self.assertEqual([_BUNDLE_ID], self.view.listdir(_ROOT))
        self.assertTrue(self.view.exists(BUNDLE_DIR))
        self.assertTrue(self.view.isdir(BUNDLE_DIR))
        self.assertEqual([_V1], self.view.listdir(BUNDLE_DIR))

        BUNDLE_DIR_V1 = join(BUNDLE_DIR, _V1)
        self.assertTrue(self.view.exists(BUNDLE_DIR_V1))
        self.assertTrue(self.view.isdir(BUNDLE_DIR_V1))
        self.assertFalse(self.view.listdir(BUNDLE_DIR_V1))

        # add a label
        BUNDLE_LABEL_FILENAME = u'bundle.xml'
        self.legacy_fs.touch(BUNDLE_LABEL_FILENAME)
        BUNDLE_LABEL_FILEPATH = join(_ROOT,
                                     _BUNDLE_ID,
                                     _V1,
                                     BUNDLE_LABEL_FILENAME)
        self.assertTrue(self.view.exists(BUNDLE_LABEL_FILEPATH))
        self.assertTrue(self.view.isfile(BUNDLE_LABEL_FILEPATH))
        self.assertEquals("", self.view.gettext(BUNDLE_LABEL_FILEPATH))

    def test_collection(self):
        """
        Test that the proper hierarchy is synthesized even for an
        empty collection in a legacy filesystem.
        """
        COLLECTION_ID = u'data_xxx_raw'
        COLLECTION_DIR = join(_ROOT, _BUNDLE_ID, COLLECTION_ID)
        self.legacy_fs.makedir(COLLECTION_ID)
        self.assertTrue(self.view.exists(COLLECTION_DIR))
        self.assertTrue(self.view.isdir(COLLECTION_DIR))
        self.assertEqual([_V1], self.view.listdir(COLLECTION_DIR))

        COLLECTION_DIR_V1 = join(COLLECTION_DIR, _V1)
        self.assertTrue(self.view.exists(COLLECTION_DIR_V1))
        self.assertTrue(self.view.isdir(COLLECTION_DIR_V1))
        self.assertFalse(self.view.listdir(COLLECTION_DIR_V1))

        # add a label
        COLLECTION_LABEL_FILENAME = u'collection.xml'
        self.legacy_fs.touch(join(COLLECTION_ID, COLLECTION_LABEL_FILENAME))
        COLLECTION_LABEL_FILEPATH = join(_ROOT,
                                         _BUNDLE_ID,
                                         COLLECTION_ID,
                                         _V1,
                                         COLLECTION_LABEL_FILENAME)
        self.assertTrue(self.view.exists(COLLECTION_LABEL_FILEPATH))
        self.assertTrue(self.view.isfile(COLLECTION_LABEL_FILEPATH))
        self.assertEquals("", self.view.gettext(COLLECTION_LABEL_FILEPATH))

    def test_product(self):
        """
        Test that the proper hierarchy is synthesized even for an
        single product in a legacy filesystem.
        """
        COLLECTION_ID = u'data_xxx_raw'
        COLLECTION_DIR = join(_ROOT, _BUNDLE_ID, COLLECTION_ID)
        VISIT = u'visit_xx'
        PRODUCT_ID = u'u2q9xx01j_raw'
        PRODUCT_DIR = join(COLLECTION_DIR, PRODUCT_ID)
        PRODUCT_DIR_V1 = join(PRODUCT_DIR, _V1)
        FITS_FILENAME = PRODUCT_ID + '.fits'
        FITS_FILEPATH = join(PRODUCT_DIR_V1, FITS_FILENAME)

        self.legacy_fs.makedirs(join(COLLECTION_ID, VISIT))
        self.legacy_fs.touch(join(COLLECTION_ID, VISIT, FITS_FILENAME))

        self.assertTrue(self.view.exists(PRODUCT_DIR))
        self.assertTrue(self.view.isdir(PRODUCT_DIR))
        self.assertEquals([_V1], self.view.listdir(PRODUCT_DIR))

        self.assertTrue(self.view.exists(PRODUCT_DIR_V1))
        self.assertTrue(self.view.isdir(PRODUCT_DIR_V1))
        self.assertEquals([FITS_FILENAME],
                          self.view.listdir(PRODUCT_DIR_V1))

        self.assertTrue(self.view.exists(FITS_FILEPATH))
        self.assertTrue(self.view.isfile(FITS_FILEPATH))

        # add a label
        PRODUCT_LABEL_FILENAME = PRODUCT_ID + '.xml'
        self.legacy_fs.touch(join(COLLECTION_ID,
                                  VISIT,
                                  PRODUCT_LABEL_FILENAME))
        PRODUCT_LABEL_FILEPATH = join(_ROOT,
                                      _BUNDLE_ID,
                                      COLLECTION_ID,
                                      PRODUCT_ID,
                                      _V1,
                                      PRODUCT_LABEL_FILENAME)
        self.assertTrue(self.view.exists(PRODUCT_LABEL_FILEPATH))
        self.assertTrue(self.view.isfile(PRODUCT_LABEL_FILEPATH))
        self.assertEquals("", self.view.gettext(PRODUCT_LABEL_FILEPATH))

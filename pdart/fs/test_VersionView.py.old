import unittest

from fs.memoryfs import MemoryFS
from fs.path import join

from pdart.fs.MultiversionBundleFS import MultiversionBundleFS
from pdart.fs.VersionView import VersionView
from pdart.fs.VersionedFS import ROOT
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

_BUNDLE_ID = u'hst_00000'
_COLLECTION_ID = u'data_xxx_raw'
_PRODUCT_ID = u'u2q9xx01j_raw'

_LIDVID_B0 = LIDVID('urn:nasa:pds:hst_00000::0.1')
_LIDVID_B1 = LIDVID('urn:nasa:pds:hst_00000::1.1')
_LIDVID_B2 = LIDVID('urn:nasa:pds:hst_00000::2.1')
_LIDVID_B3 = LIDVID('urn:nasa:pds:hst_00000::3.1')

_LIDVID_C0 = LIDVID('urn:nasa:pds:hst_00000:data_xxx_raw::0.1')
_LIDVID_C1 = LIDVID('urn:nasa:pds:hst_00000:data_xxx_raw::1.1')
_LIDVID_C2 = LIDVID('urn:nasa:pds:hst_00000:data_xxx_raw::2.1')

_LIDVID_P0 = LIDVID('urn:nasa:pds:hst_00000:data_xxx_raw:u2q9xx01j_raw::0.1')
_LIDVID_P1 = LIDVID('urn:nasa:pds:hst_00000:data_xxx_raw:u2q9xx01j_raw::1.1')


class TestVersionView(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        memory_fs = MemoryFS()
        self.versioned_fs = MultiversionBundleFS(memory_fs)
        self.versioned_fs.make_lidvid_directories(_LIDVID_B0)
        self.versioned_fs.make_lidvid_directories(_LIDVID_B1)
        self.versioned_fs.make_lidvid_directories(_LIDVID_B2)
        self.versioned_fs.make_lidvid_directories(_LIDVID_B3)

        self.versioned_fs.make_lidvid_directories(_LIDVID_C0)
        self.versioned_fs.make_lidvid_directories(_LIDVID_C1)
        self.versioned_fs.add_subcomponent(_LIDVID_B3, _LIDVID_C2)

        self.versioned_fs.make_lidvid_directories(_LIDVID_P0)
        self.versioned_fs.add_subcomponent(_LIDVID_C2, _LIDVID_P1)

        self.version_view = VersionView(_LIDVID_B3,
                                        self.versioned_fs)

    @unittest.skip('fails on purpose; for use while testing')
    def test_init(self):
        # type: () -> None
        self.versioned_fs.tree()
        self.assertFalse(True)

    def test_creation(self):
        # type: () -> None
        self.assertEqual(self.version_view._bundle_id, _BUNDLE_ID)
        self.assertEqual(self.version_view._version_id, u'3.1')
        with self.assertRaises(Exception):
            VersionView(LIDVID('urn:nasa:pds:hst_00000::666.333'),
                        self.versioned_fs)

    def test_root(self):
        # type: () -> None
        self.assertTrue(self.version_view.exists(ROOT))
        self.assertTrue(self.version_view.isdir(ROOT))
        self.assertEqual([_BUNDLE_ID], self.version_view.listdir(ROOT))

    def test_bundle_dir(self):
        # type: () -> None
        BUNDLE_DIR = join(ROOT, _BUNDLE_ID)
        self.assertTrue(self.version_view.exists(BUNDLE_DIR))
        self.assertTrue(self.version_view.isdir(BUNDLE_DIR))
        self.assertEqual([_COLLECTION_ID],
                         self.version_view.listdir(BUNDLE_DIR))

        # test that collections appear
        self.assertTrue(self.version_view.exists(
            join(ROOT, _BUNDLE_ID, _COLLECTION_ID)))

        # test that files don't appear when wrong version
        self.versioned_fs.touch(
            join(ROOT, _BUNDLE_ID, u'v$1.1', u'bundle.xml'))
        self.assertFalse(self.version_view.exists(
            join(ROOT, _BUNDLE_ID, u'bundle.xml')))

        # test that files do appear when right version
        self.versioned_fs.touch(
            join(ROOT, _BUNDLE_ID, u'v$3.1', u'bundle.xml'))
        self.assertTrue(self.version_view.exists(
            join(ROOT, _BUNDLE_ID, u'bundle.xml')))

    def test_collection_dir(self):
        # type: () -> None
        COLLECTION_DIR = join(ROOT, _BUNDLE_ID, _COLLECTION_ID)

        self.assertTrue(self.version_view.exists(COLLECTION_DIR))
        self.assertTrue(self.version_view.isdir(COLLECTION_DIR))
        self.assertEqual([_PRODUCT_ID],
                         self.version_view.listdir(COLLECTION_DIR))

        # test that files don't appear when wrong version
        self.versioned_fs.touch(join(COLLECTION_DIR, u'v$1.1',
                                     u'collection.xml'))
        self.assertFalse(self.version_view.exists(join(COLLECTION_DIR,
                                                       u'collection.xml')))
        # test that files do appear when right version
        self.versioned_fs.touch(join(COLLECTION_DIR, u'v$2.1',
                                     u'collection.xml'))
        self.assertTrue(self.version_view.exists(
            join(COLLECTION_DIR, u'collection.xml')))

    def test_product_dir(self):
        # type: () -> None
        PRODUCT_DIR = join(ROOT, _BUNDLE_ID, _COLLECTION_ID, _PRODUCT_ID)
        self.assertTrue(self.version_view.exists(PRODUCT_DIR))
        self.assertTrue(self.version_view.isdir(PRODUCT_DIR))
        self.assertEqual([], self.version_view.listdir(PRODUCT_DIR))

        # test that files don't appear when wrong version
        self.versioned_fs.touch(join(PRODUCT_DIR, u'v$0.1',
                                     u'product.xml'))
        self.assertFalse(self.version_view.exists(join(PRODUCT_DIR,
                                                       u'product.xml')))

        # test that files do appear when right version
        self.versioned_fs.touch(join(PRODUCT_DIR, u'v$1.1',
                                     u'product.xml'))
        self.assertTrue(self.version_view.exists(
            join(PRODUCT_DIR, u'product.xml')))

    def test_directory_to_lid(self):
        # type: () -> None
        with self.assertRaises(AssertionError):
            VersionView.directory_to_lid(u'/')
        self.assertEqual(LID('urn:nasa:pds:b'),
                         VersionView.directory_to_lid(u'/b'))
        self.assertEqual(LID('urn:nasa:pds:b:c'),
                         VersionView.directory_to_lid(u'/b/c/'))
        self.assertEqual(LID('urn:nasa:pds:b:c:p'),
                         VersionView.directory_to_lid(u'/b/c/p'))

        # what if wrong kind of directory?  i.e., from MultiversionBundleFS?
        with self.assertRaises(AssertionError):
            VersionView.directory_to_lid(u'/b/c/v$23.1')

    def test_lid_to_vid(self):
        with self.assertRaises(Exception):
            self.version_view.lid_to_vid(LID(u'urn:nasa:pds:b'))
        self.versioned_fs.tree()
        self.assertEqual(
            VID('3.1'),
            self.version_view.lid_to_vid(LID(u'urn:nasa:pds:hst_00000')))
        self.assertEqual(
            VID('2.1'),
            self.version_view.lid_to_vid(
                LID(u'urn:nasa:pds:hst_00000:data_xxx_raw')))
        self.assertEqual(
            VID('1.1'),
            self.version_view.lid_to_vid(
                LID(u'urn:nasa:pds:hst_00000:data_xxx_raw:u2q9xx01j_raw')))

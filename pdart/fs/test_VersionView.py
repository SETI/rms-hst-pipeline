from pdart.fs.VersionView import VersionView

import unittest

from fs.memoryfs import MemoryFS
from fs.path import join

from pdart.fs.SubdirVersions import writeSubdirVersions
from pdart.fs.VersionedFS import ROOT

_BUNDLE_ID = u'hst_00000'
_COLLECTION_ID = u'data_xxx_raw'
_PRODUCT_ID = u'u2q9xx01j_raw'


class TestVersionView(unittest.TestCase):
    def setUp(self):
        self.versioned_fs = MemoryFS()
        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v$0'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, u'v$0'), {})

        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v$1'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, u'v$1'), {})

        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v$2'))

        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v$3'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, u'v$3'),
                            {_COLLECTION_ID: u'2'})

        self.versioned_fs.makedirs(
                join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$0'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$0'),
                            {})

        self.versioned_fs.makedirs(
                join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$1'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$1'),
                            {})

        self.versioned_fs.makedirs(
                join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$2'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$2'),
                            {_PRODUCT_ID: u'1'})

        self.versioned_fs.makedirs(
            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, _PRODUCT_ID, u'v$0'))

        self.versioned_fs.makedirs(
            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, _PRODUCT_ID, u'v$1'))

        self.version_view = VersionView(u'urn:nasa:pds:hst_00000::3',
                                        self.versioned_fs)

    def test_creation(self):
        # type: () -> None
        self.assertEqual(self.version_view._bundle_id, _BUNDLE_ID)
        self.assertEqual(self.version_view._version_id, u'3')
        with self.assertRaises(Exception):
            VersionView(u'urn:nasa:pds:hst_00000::666', self.versioned_fs)

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
        self.versioned_fs.touch(join(ROOT, _BUNDLE_ID, u'v$1', u'bundle.xml'))
        self.assertFalse(self.version_view.exists(
                join(ROOT, _BUNDLE_ID, u'bundle.xml')))

        # test that files do appear when right version
        self.versioned_fs.touch(join(ROOT, _BUNDLE_ID, u'v$3', u'bundle.xml'))
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
        self.versioned_fs.touch(join(COLLECTION_DIR, u'v$1',
                                     u'collection.xml'))
        self.assertFalse(self.version_view.exists(join(COLLECTION_DIR,
                                                       u'collection.xml')))
        # test that files do appear when right version
        self.versioned_fs.touch(join(COLLECTION_DIR, u'v$2',
                                     u'collection.xml'))
        print '****', self.version_view.listdir(COLLECTION_DIR)
        print '****', self.version_view.getinfo(join(COLLECTION_DIR,
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
        self.versioned_fs.touch(join(PRODUCT_DIR, u'v$0',
                                     u'product.xml'))
        self.assertFalse(self.version_view.exists(join(PRODUCT_DIR,
                                                       u'product.xml')))

        # test that files do appear when right version
        self.versioned_fs.touch(join(PRODUCT_DIR, u'v$1',
                                     u'product.xml'))
        self.assertTrue(self.version_view.exists(
                join(PRODUCT_DIR, u'product.xml')))

    def test_legacy_bundle_dir(self):
        BUNDLE_DIR = join(ROOT, _BUNDLE_ID, u'v$3')
        self.assertEqual(BUNDLE_DIR, self.version_view._legacy_bundle_dir())

    def test_legacy_collection_dir(self):
        COLLECTION_DIR = join(ROOT, _BUNDLE_ID, _COLLECTION_ID, u'v$2')
        legacy_dir = self.version_view._legacy_collection_dir(_COLLECTION_ID)
        self.assertEqual(COLLECTION_DIR, legacy_dir)

    def test_legacy_product_dir(self):
        PRODUCT_DIR = join(ROOT,
                           _BUNDLE_ID, _COLLECTION_ID,
                           _PRODUCT_ID, u'v$1')
        self.assertEqual(PRODUCT_DIR,
                         self.version_view._legacy_product_dir(_COLLECTION_ID,
                                                               _PRODUCT_ID))

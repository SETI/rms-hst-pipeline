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
        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v1'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, u'v1'), {})

        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v2'))

        self.versioned_fs.makedirs(join(ROOT, _BUNDLE_ID, u'v3'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, u'v3'),
                            {_COLLECTION_ID: u'v2'})

        self.versioned_fs.makedirs(
                join(ROOT, _BUNDLE_ID, _COLLECTION_ID, 'v1'))
        self.versioned_fs.makedirs(
                join(ROOT, _BUNDLE_ID, _COLLECTION_ID, 'v2'))
        writeSubdirVersions(self.versioned_fs,
                            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, 'v2'),
                            {_PRODUCT_ID: u'v1'})

        self.versioned_fs.makedirs(
            join(ROOT, _BUNDLE_ID, _COLLECTION_ID, _PRODUCT_ID, u'v1'))

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
        self.assertEqual([u'data_xxx_raw'],
                         self.version_view.listdir(BUNDLE_DIR))

        # test that files appear
        self.versioned_fs.touch(join(ROOT, _BUNDLE_ID, u'v1', u'bundle.xml'))
        self.assertFalse(self.version_view.exists(
                join(ROOT, _BUNDLE_ID, u'bundle.xml')))
        self.versioned_fs.touch(join(ROOT, _BUNDLE_ID, u'v3', u'bundle.xml'))
        self.assertTrue(self.version_view.exists(
                join(ROOT, _BUNDLE_ID, u'bundle.xml')))

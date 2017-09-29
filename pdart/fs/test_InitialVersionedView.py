import unittest

import fs.copy
from fs.memoryfs import MemoryFS

from pdart.fs.InitialVersionedView import *
from pdart.fs.VersionedViewVerifier import *
from pdart.pds4.Archives import get_any_archive

_BUNDLE_ID = u'hst_00000'


# type: unicode


class TestInitialVersionedView(unittest.TestCase):
    def setUp(self):
        self.legacy_fs = MemoryFS()
        self.bundle_id = _BUNDLE_ID
        self.view = InitialVersionedView(self.bundle_id, self.legacy_fs)

    def test_bare_fs(self):
        """
        Test that the proper hierarchy is synthesized even for an
        empty legacy filesystem.
        """
        BUNDLE_DIR = join(ROOT, _BUNDLE_ID)
        self.assertTrue(self.view.exists(ROOT))
        self.assertTrue(self.view.isdir(ROOT))
        self.assertEqual([_BUNDLE_ID], self.view.listdir(ROOT))
        self.assertTrue(self.view.exists(BUNDLE_DIR))
        self.assertTrue(self.view.isdir(BUNDLE_DIR))
        self.assertEqual([VERSION_ONE], self.view.listdir(BUNDLE_DIR))

        BUNDLE_DIR_V1 = join(BUNDLE_DIR, VERSION_ONE)
        self.assertTrue(self.view.exists(BUNDLE_DIR_V1))
        self.assertTrue(self.view.isdir(BUNDLE_DIR_V1))
        self.assertEqual([SUBDIR_VERSIONS_FILENAME],
                         self.view.listdir(BUNDLE_DIR_V1))

        # add a label
        BUNDLE_LABEL_FILENAME = u'bundle.xml'
        self.legacy_fs.touch(BUNDLE_LABEL_FILENAME)
        BUNDLE_LABEL_FILEPATH = join(ROOT,
                                     _BUNDLE_ID,
                                     VERSION_ONE,
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
        COLLECTION_DIR = join(ROOT, _BUNDLE_ID, COLLECTION_ID)
        self.legacy_fs.makedir(COLLECTION_ID)
        self.assertTrue(self.view.exists(COLLECTION_DIR))
        self.assertTrue(self.view.isdir(COLLECTION_DIR))
        self.assertEqual([VERSION_ONE], self.view.listdir(COLLECTION_DIR))

        COLLECTION_DIR_V1 = join(COLLECTION_DIR, VERSION_ONE)
        self.assertTrue(self.view.exists(COLLECTION_DIR_V1))
        self.assertTrue(self.view.isdir(COLLECTION_DIR_V1))
        self.assertEqual([SUBDIR_VERSIONS_FILENAME],
                         self.view.listdir(COLLECTION_DIR_V1))

        # add a label
        COLLECTION_LABEL_FILENAME = u'collection.xml'
        self.legacy_fs.touch(join(COLLECTION_ID, COLLECTION_LABEL_FILENAME))
        COLLECTION_LABEL_FILEPATH = join(ROOT,
                                         _BUNDLE_ID,
                                         COLLECTION_ID,
                                         VERSION_ONE,
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
        COLLECTION_DIR = join(ROOT, _BUNDLE_ID, COLLECTION_ID)
        VISIT = u'visit_xx'
        PRODUCT_ID = u'u2q9xx01j_raw'
        PRODUCT_DIR = join(COLLECTION_DIR, PRODUCT_ID)
        PRODUCT_DIR_V1 = join(PRODUCT_DIR, VERSION_ONE)
        FITS_FILENAME = PRODUCT_ID + '.fits'
        FITS_FILEPATH = join(PRODUCT_DIR_V1, FITS_FILENAME)

        self.legacy_fs.makedirs(join(COLLECTION_ID, VISIT))
        self.legacy_fs.touch(join(COLLECTION_ID, VISIT, FITS_FILENAME))

        self.assertTrue(self.view.exists(PRODUCT_DIR))
        self.assertTrue(self.view.isdir(PRODUCT_DIR))
        self.assertEquals([VERSION_ONE], self.view.listdir(PRODUCT_DIR))

        self.assertTrue(self.view.exists(PRODUCT_DIR_V1))
        self.assertTrue(self.view.isdir(PRODUCT_DIR_V1))
        self.assertEquals(set([FITS_FILENAME, SUBDIR_VERSIONS_FILENAME]),
                          set(self.view.listdir(PRODUCT_DIR_V1)))

        self.assertTrue(self.view.exists(FITS_FILEPATH))
        self.assertTrue(self.view.isfile(FITS_FILEPATH))

        # add a label
        PRODUCT_LABEL_FILENAME = PRODUCT_ID + '.xml'
        self.legacy_fs.touch(join(COLLECTION_ID,
                                  VISIT,
                                  PRODUCT_LABEL_FILENAME))
        PRODUCT_LABEL_FILEPATH = join(ROOT,
                                      _BUNDLE_ID,
                                      COLLECTION_ID,
                                      PRODUCT_ID,
                                      VERSION_ONE,
                                      PRODUCT_LABEL_FILENAME)
        self.assertTrue(self.view.exists(PRODUCT_LABEL_FILEPATH))
        self.assertTrue(self.view.isfile(PRODUCT_LABEL_FILEPATH))
        self.assertEquals("", self.view.gettext(PRODUCT_LABEL_FILEPATH))


def test_initial_versioned_view_as_versioned_view():
    # type: () -> None
    with MemoryFS() as memoryFS:
        memoryFS.makedirs(u'/data_xxx_raw/visit_xx')
        memoryFS.touch(u'/data_xxx_raw/visit_xx/u2q9xx01j_raw.fits')
        with InitialVersionedView(_BUNDLE_ID, memoryFS) as fs:
            InitialVersionedViewVerifier(fs)


class InitialVersionedViewVerifier(VersionedViewVerifier):
    def check_subdir_versions_file(self,
                                   version_dir):
        # call the superclass's version for standard tests
        VersionedViewVerifier.check_subdir_versions_file(self, version_dir)

        # In a filesystem with only one version, we have one
        # additional condition: all subdirectories in the filesystem
        # should appear in the subdir_versions file.
        (ordinary_file_infos,
         ordinary_dir_infos,
         subdir_versions_file_infos,
         version_dir_infos) = scan_vfs_dir(self.view, join(version_dir, '..'))
        expected = set(info.name for info in ordinary_dir_infos)

        actual = set(read_subdir_versions(self.view, version_dir).keys())

        self.assertEqual(expected, actual)


@unittest.skip('takes a long time')
def test_initial_versioned_view_on_archive():
    """
    Run through all the bundles in the archive, view them as versioned
    filesystems, and try to verify them , then copy them to another
    (in-memory) filesystem.  See whether anything breaks.
    """
    archive = get_any_archive()
    for bundle in archive.bundles():
        print bundle
        with OSFS(bundle.absolute_filepath()) as osfs:
            view = InitialVersionedView(bundle.lid.bundle_id, osfs)
            InitialVersionedViewVerifier(view)
            with MemoryFS() as memoryfs:
                fs.copy.copy_fs(view, memoryfs)

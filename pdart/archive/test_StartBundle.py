import os
import shutil
import tempfile
import unittest

import fs.path

from pdart.archive.StartBundle import *
from pdart.archive.StartBundle import _INITIAL_VID
from pdart.fs.V1FS import _V1_0
from pdart.fs.VersionedFS import SUBDIR_VERSIONS_FILENAME
from pdart.new_db.BundleDB import _BUNDLE_DB_NAME


def _path_to_testfiles():
    # type: () -> unicode
    """Return the path to files needed for testing."""
    return os.path.join(os.path.dirname(__file__), 'testfiles')


def _list_rel_filepaths(root_dir):
    # type: (unicode) -> List[unicode]
    def _list_rel_filepaths_gen():
        for (dirpath, _, filenames) in os.walk(root_dir):
            rel_dirpath = fs.path.relativefrom(root_dir, dirpath)
            for filename in filenames:
                yield os.path.join(rel_dirpath, filename)

    return sorted(_list_rel_filepaths_gen())


class TestStartBundle(unittest.TestCase):
    def setUp(self):
        self.base_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base_directory)

    def test_bundle_to_int(self):
        self.assertEqual(bundle_to_int('hst_01234'), 1234)
        self.assertFalse(bundle_to_int('george'))
        self.assertFalse(bundle_to_int('hst_0000'))
        self.assertFalse(bundle_to_int('hst_000000'))

    def test_copy_files_from_download(self):
        # type: () -> None
        download_dir = _path_to_testfiles()
        res = copy_files_from_download(download_dir, self.base_directory)
        # check that it read the expected bundle
        self.assertEqual(res, 13012)

        # Check that the newly created archive dir is in multi-version
        # format.
        osfs = OSFS(self.base_directory)
        BUNDLE_DIR = u'/hst_13012'
        self.assertTrue(osfs.exists(BUNDLE_DIR))

        BUNDLE_VERSION_1_DIR = fs.path.join(BUNDLE_DIR, _V1_0)
        self.assertTrue(osfs.exists(BUNDLE_VERSION_1_DIR))

        SUBDIR_VERSIONS = fs.path.join(BUNDLE_DIR, _V1_0,
                                       SUBDIR_VERSIONS_FILENAME)
        self.assertTrue(osfs.exists(SUBDIR_VERSIONS))

    def test_create_bundle_db(self):
        # type: () -> None
        download_dir = _path_to_testfiles()
        copy_files_from_download(download_dir, self.base_directory)
        db = create_bundle_db(13012, self.base_directory)
        try:
            # returns the DB
            self.assertTrue(db)
            db_filename = os.path.join(self.base_directory,
                                       'hst_13012',
                                       _BUNDLE_DB_NAME)
            # creates the DB file
            self.assertTrue(os.path.isfile(db_filename))

            bundle_lid = LID.create_from_parts(['hst_13012'])
            bundle_lidvid = LIDVID.create_from_lid_and_vid(
                bundle_lid,
                _INITIAL_VID)
            self.assertTrue(db.bundle_exists(str(bundle_lidvid)))
        finally:
            db.close()

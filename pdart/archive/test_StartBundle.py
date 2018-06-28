import os.path
import tempfile
import unittest

import fs.path

from pdart.archive.StartBundle import *


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

    def test_create_bundle_dir(self):
        # make one; test that it creates its directory
        bundle_dir = os.path.join(self.base_directory, 'hst_01234')
        self.assertFalse(os.path.isdir(bundle_dir))
        create_bundle_dir(1234, self.base_directory)
        self.assertTrue(os.path.isdir(bundle_dir))

        # make another; test that it creates its directory
        bundle_dir = os.path.join(self.base_directory, 'hst_00000')
        self.assertFalse(os.path.isdir(bundle_dir))
        create_bundle_dir(0, self.base_directory)
        self.assertTrue(os.path.isdir(bundle_dir))

        # re-making does nothing and is harmless
        bundle_dir = os.path.join(self.base_directory, 'hst_00000')
        self.assertTrue(os.path.isdir(bundle_dir))
        create_bundle_dir(0, self.base_directory)
        self.assertTrue(os.path.isdir(bundle_dir))

        # raises an exception if its a file: just shouldn't happen
        bundle_dir = os.path.join(self.base_directory, 'hst_00666')
        self.assertFalse(os.path.isdir(bundle_dir))
        self.assertFalse(os.path.isfile(bundle_dir))
        with open(bundle_dir, 'w') as f:
            f.write('xxx')

        self.assertTrue(os.path.isfile(bundle_dir))
        with self.assertRaises(Exception):
            create_bundle_dir(666, self.base_directory)

    def test_create_bundle_db(self):
        bundle_id = 12345
        create_bundle_dir(bundle_id, self.base_directory)
        db = create_bundle_db(bundle_id, self.base_directory)
        try:
            # returns the DB
            self.assertTrue(db)
            db_filename = os.path.join(self.base_directory,
                                       'hst_12345',
                                       'bundle$database.db')
            # creates the DB file
            self.assertTrue(os.path.isfile(db_filename))
            self.assertEquals([
                u'hst_12345/bundle$database.db'
            ], _list_rel_filepaths(self.base_directory))

            # TODO Could check for creation of tables, though
            # existence of the file shows that the file was written
            # to.
        finally:
            db.close()

    def test_copy_downloaded_files(self):
        # (bundle_id, download_root, archive_dir):
        bundle_id = 13012
        create_bundle_dir(bundle_id, self.base_directory)
        create_bundle_db(bundle_id, self.base_directory)

        copy_downloaded_files(bundle_id, _path_to_testfiles(),
                              self.base_directory)
        self.assertEquals([
            u'hst_13012/bundle$database.db',
            u'hst_13012/data_acs_drz/jbz504010/v$1/jbz504011_drz.fits',
            u'hst_13012/data_acs_drz/jbz504020/v$1/jbz504021_drz.fits',
            u'hst_13012/data_acs_drz/jbz504eoq/v$1/jbz504eoq_drz.fits',
            u'hst_13012/data_acs_flt/jbz504eoq/v$1/jbz504eoq_flt.fits'],
            _list_rel_filepaths(self.base_directory))

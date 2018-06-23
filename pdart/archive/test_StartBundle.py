import os.path
import unittest

from pdart.archive.StartBundle import *


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
        create_bundle_dir(12345, self.base_directory)
        db = create_bundle_db(12345, self.base_directory)
        try:
            # returns the DB
            self.assertTrue(db)
            db_filename = os.path.join(self.base_directory,
                                       'hst_12345',
                                       'bundle$database.db')
            # creates the DB file
            self.assertTrue(os.path.isfile(db_filename))

            # TODO should check for creation of tables, though
            # existence of the file shows that the file was written
            # to.
        finally:
            db.close()

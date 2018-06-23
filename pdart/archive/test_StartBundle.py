import os.path
import shutil
import tempfile
import unittest

from pdart.archive.StartBundle import create_bundle_dir

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

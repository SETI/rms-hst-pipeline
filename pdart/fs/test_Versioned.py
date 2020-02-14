import shutil
import tempfile
import unittest
from fs.path import join
from pdart.fs.Versioned import *

class Test_Versioned(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_single_versioned_osfs(self):
        partial_path = join(self.temp_dir, 'henrietta')
        fs = SingleVersionedOSFS.create_suffixed(partial_path, create=True)
        self.assertTrue(fs.is_single_versioned_fs())
        names = OSFS(self.temp_dir).listdir(u'/')
        self.assertEquals(['henrietta-sv'], names)

    def test_multiversioned_osfs(self):
        partial_path = join(self.temp_dir, 'henrietta')
        fs = MultiversionedOSFS.create_suffixed(partial_path, create=True)
        self.assertTrue(fs.is_multiversioned_fs())
        names = OSFS(self.temp_dir).listdir(u'/')
        self.assertEquals(['henrietta-mv'], names)

    def test_single_versioned_cowfs(self):
        base_partial_path = join(self.temp_dir, 'base')
        fs = SingleVersionedOSFS.create_suffixed(base_partial_path,
                                                 create=True)
        rw_partial_path = join(self.temp_dir, u'next')
        cowfs = SingleVersionedCOWFS.create_cowfs_suffixed(
            fs,
            rw_partial_path,
            recreate=True)
        self.assertTrue(cowfs.is_single_versioned_fs())
        names = OSFS(self.temp_dir).listdir(u'/')
        self.assertEquals({u'base-sv', u'next-deltas-layer-sv'},
                          set(names))

    def test_multiversioned_cowfs(self):
        base_partial_path = join(self.temp_dir, 'base')
        fs = MultiversionedOSFS.create_suffixed(base_partial_path,
                                                 create=True)
        rw_partial_path = join(self.temp_dir, u'next')
        cowfs = MultiversionedCOWFS.create_cowfs_suffixed(
            fs,
            rw_partial_path,
            recreate=True)
        self.assertTrue(cowfs.is_multiversioned_fs())
        names = OSFS(self.temp_dir).listdir(u'/')
        self.assertEquals({u'base-mv', u'next-deltas-layer-mv'},
                          set(names))

import shutil
import tempfile
import unittest

from fs.path import join

from pdart.fs.versioned.Versioned import *


class Test_Versioned(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_single_versioned_osfs(self) -> None:
        partial_path = join(self.temp_dir, "henrietta")
        fs = SingleVersionedOSFS.create_suffixed(partial_path, create=True)
        self.assertTrue(fs.is_single_versioned_fs())
        names = OSFS(self.temp_dir).listdir("/")
        self.assertEqual(["henrietta-sv"], names)

    def test_multiversioned_osfs(self) -> None:
        partial_path = join(self.temp_dir, "henrietta")
        fs = MultiversionedOSFS.create_suffixed(partial_path, create=True)
        self.assertTrue(fs.is_multiversioned_fs())
        names = OSFS(self.temp_dir).listdir("/")
        self.assertEqual(["henrietta-mv"], names)

    def test_single_versioned_cowfs(self) -> None:
        base_partial_path = join(self.temp_dir, "base")
        fs = SingleVersionedOSFS.create_suffixed(base_partial_path, create=True)
        rw_partial_path = join(self.temp_dir, "next")
        cowfs = SingleVersionedCOWFS.create_cowfs_suffixed(
            fs, rw_partial_path, recreate=True
        )
        self.assertTrue(cowfs.is_single_versioned_fs())
        names = OSFS(self.temp_dir).listdir("/")
        self.assertEqual({"base-sv", "next-deltas-sv"}, set(names))

    def test_multiversioned_cowfs(self) -> None:
        base_partial_path = join(self.temp_dir, "base")
        fs = MultiversionedOSFS.create_suffixed(base_partial_path, create=True)
        rw_partial_path = join(self.temp_dir, "next")
        cowfs = MultiversionedCOWFS.create_cowfs_suffixed(
            fs, rw_partial_path, recreate=True
        )
        self.assertTrue(cowfs.is_multiversioned_fs())
        names = OSFS(self.temp_dir).listdir("/")
        self.assertEqual({"base-mv", "next-deltas-mv"}, set(names))

import unittest

from fs.memoryfs import MemoryFS

from pdart.pipeline.fs_types import *


class TestFSTypes(unittest.TestCase):
    def test_is_version_dir(self) -> None:
        self.assertFalse(is_version_dir("/foo/bar"))
        self.assertFalse(is_version_dir("/foo/bar/v$3"))
        self.assertTrue(is_version_dir("/foo/bar/v$312.345"))
        self.assertFalse(is_version_dir("/foo/bar/v$$312.345"))

    def test_categorize_filesystem(self) -> None:
        m = MemoryFS()
        self.assertEqual(EMPTY_FS_TYPE, categorize_filesystem(m))

        m = MemoryFS()
        m.makedir("/hst_12345$")
        self.assertEqual(SINGLE_VERSIONED_FS_TYPE, categorize_filesystem(m))

        m = MemoryFS()
        m.makedirs("/hst_12345/v$1.0")
        self.assertEqual(MULTIVERSIONED_FS_TYPE, categorize_filesystem(m))

        m = MemoryFS()
        m.makedir("/hst_12345")
        self.assertEqual(UNKNOWN_FS_TYPE, categorize_filesystem(m))

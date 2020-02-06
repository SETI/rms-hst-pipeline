import unittest
from fs.memoryfs import MemoryFS
from pdart.pipeline.FSTypes import *

class Test_FSTypes(unittest.TestCase):
    def test_is_version_dir(self):
        # type: () -> None
        self.assertFalse(is_version_dir('/foo/bar'))
        self.assertFalse(is_version_dir('/foo/bar/v$3'))
        self.assertTrue(is_version_dir('/foo/bar/v$312.345'))
        self.assertFalse(is_version_dir('/foo/bar/v$$312.345'))

    def test_categorize_filesystem(self):
        # type: () -> None
        m = MemoryFS()
        self.assertEqual(EMPTY_FS_TYPE, categorize_filesystem(m))

        m = MemoryFS()
        m.makedir(u'/hst_12345$')
        self.assertEqual(SINGLE_VERSIONED_FS_TYPE, categorize_filesystem(m))
        
        m = MemoryFS()
        m.makedirs(u'/hst_12345/v$1.0')
        self.assertEqual(MULTIVERSIONED_FS_TYPE, categorize_filesystem(m))
        
        m = MemoryFS()
        m.makedir(u'/hst_12345')
        self.assertEqual(UNKNOWN_FS_TYPE, categorize_filesystem(m))
        

from pdart.db.DatabaseArchive import DatabaseArchive

import tempfile
import unittest

from fs.path import dirname


class TestDatabaseArchive(unittest.TestCase):
    def test_DatabaseArchive(self):
        # type: () -> None
        # Fails with no file
        with self.assertRaises(Exception):
            DatabaseArchive(None)

        # Succeeds with filepath.
        filepath = tempfile.mktemp()
        da = DatabaseArchive(filepath)
        self.assertTrue(da.is_open())
        da.close()
        self.assertFalse(da.is_open())

        # Verify archive_dir
        self.assertEquals(dirname(filepath), da.archive_dir)

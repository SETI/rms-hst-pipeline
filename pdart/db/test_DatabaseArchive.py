from pdart.db.DatabaseArchive import *

from contextlib import closing
import os.path
import tempfile
import unittest

from pdart.pds4.Archives import *


class TestDatabaseArchive(unittest.TestCase):
    def test_DatabaseArchive(self):
        # Calling DatabaseArchive() will either open a file or create
        # a new one.  For that reason, I should probably use a tmpfile
        # instead.
        filepath = tempfile.mktemp()
        da = DatabaseArchive(filepath)
        self.assertTrue(da.is_open())
        da.close()
        self.assertFalse(da.is_open())

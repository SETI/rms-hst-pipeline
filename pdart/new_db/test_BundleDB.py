import os
import tempfile
import unittest

from BundleDB import *


class Test_BundleDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        (_, self.db_filepath) = tempfile.mkstemp(suffix='.db')

    def tearDown(self):
        # type: () -> None
        os.remove(self.db_filepath)

    def test_BundleDB(self):
        # type: () -> None
        db = BundleDB(self.db_filepath)
        self.assertTrue(db.is_open())
        db.close()
        self.assertFalse(db.is_open())

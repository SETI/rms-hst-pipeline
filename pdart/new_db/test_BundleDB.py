import os
import tempfile
import unittest

from pdart.new_db.BundleDB import *
from pdart.new_db.SqlAlchTables import Base

if TYPE_CHECKING:
    from typing import List

_TABLES = ['bundles']  # type: List[str]


class Test_BundleDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        (_, self.db_os_filepath) = tempfile.mkstemp(suffix='.db')
        self.db = BundleDB(self.db_os_filepath)

    def tearDown(self):
        # type: () -> None
        os.remove(self.db_os_filepath)

    def test_BundleDB(self):
        # type: () -> None
        db = BundleDB(self.db_os_filepath)
        self.assertTrue(db.is_open())
        db.create_tables()
        metadata = Base.metadata
        self.assertEqual(metadata.tables.keys(), _TABLES)
        db.close()
        self.assertFalse(db.is_open())

import os
import tempfile
import unittest

from pdart.new_db.BundleDB import *
from pdart.new_db.SqlAlchTables import Base

if TYPE_CHECKING:
    from typing import Set

_TABLES = {'bundles',
           'collections', 'document_collections', 'non_document_collections',
           'products', 'browse_products', 'document_products', 'fits_products',
           'document_files', 'hdus', 'cards'}  # type: Set[str]


class Test_BundleDB(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = BundleDB.create_database_in_memory()
        self.db.create_tables()

    def test_BundleDB(self):
        # type: () -> None
        db = self.db
        self.assertTrue(db.is_open())
        metadata = Base.metadata
        self.assertEqual(set(metadata.tables.keys()), _TABLES)
        db.close()
        self.assertFalse(db.is_open())

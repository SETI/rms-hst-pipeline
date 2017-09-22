from pdart.db.CompleteDatabase import exists_database_records_for_browse, \
    exists_database_records_for_fits, insert_browse_database_records, \
    insert_fits_database_records

from contextlib import closing
from os import remove
import sqlite3
import tempfile
import unittest

from fs.path import join

from pdart.pds4.LID import LID


class TestCompleteDatabase(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db_name = join(tempfile.gettempdir(), 'test.db')
        self.database_conn = sqlite3.connect(self.db_name)

    def tearDown(self):
        # type: () -> None
        if self.database_conn:
            self.database_conn.close()
            try:
                remove(self.db_name)
            except OSError:
                pass

    def test_insert_fits_database_records(self):
        """
        Tests that insert_fits_database_records() does indeed do that
        (assuming exists_database_records_for_fits() is correctly
        implemented).
        """
        # type: () -> None
        with closing(self.database_conn.cursor()) as cursor:
            lid = LID('urn:nasa:pds:bundle:container:product')
            self.assertFalse(exists_database_records_for_fits(
                    self.database_conn, lid))
            insert_fits_database_records(cursor, lid)
            self.assertTrue(exists_database_records_for_fits(
                    self.database_conn, lid))

    def test_insert_browse_database_records(self):
        """
        Tests that insert_browse_database_records() does indeed do that
        (assuming exists_database_records_for_browse() is correctly
        implemented).
        """
        # type: () -> None
        with closing(self.database_conn.cursor()) as cursor:
            lid = LID('urn:nasa:pds:bundle:container:browse_product')
            self.assertFalse(exists_database_records_for_browse(
                    self.database_conn, lid))
            insert_browse_database_records(cursor, lid)
            self.assertTrue(exists_database_records_for_browse(
                    self.database_conn, lid))

"""
This module creates a SQLite database from a bundle.
"""
import os.path

from pdart.db.CreateDatabase import DatabaseCreator
from pdart.db.TableSchemas import *
from pdart.pds4.Bundle import Bundle
import pyfits
import sqlite3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pd4.Archive


def _create_bundles_table(conn, bundle):
    # type: (sqlite3.Connection, Bundle) -> None
    """Create the bundles table."""
    conn.execute('DROP TABLE IF EXISTS bundles')
    conn.execute(BUNDLES_SCHEMA)
    conn.execute(BUNDLES_SQL, bundle_tuple(bundle))
    conn.commit()


def create_bundle_database(conn, bundle):
    # type: (sqlite3.Connection, Bundle) -> None
    """
    Given an open SQLite connection to a fresh database and an
    :class:`~pdart.pds4.Archive`, populate the database with the
    archive's information.
    """
    conn.execute('PRAGMA foreign_keys = ON;')
    _create_bundles_table(conn, bundle)


class BundleDatabaseCreator(DatabaseCreator):
    def __init__(self, archive):
        # type: (pdart.pds4.Archive.Archive) -> None
        DatabaseCreator.__init__(self)
        self.archive = archive
        self.conn = None
        # type: sqlite3.Connection
        self.bundle = None
        # type: Bundle

    def get_conn(self):
        return self.conn

    def create(self):
        """
        Given an open SQLite connection to a fresh database and an
        :class:`~pdart.pds4.Archive`, populate the database with the
        archive's information.
        """
        for b in self.archive.bundles():
            self.conn = open_bundle_database(self.bundle)
            try:
                self.conn.execute('PRAGMA foreign_keys = ON;')
                self.create_bundles_table()
                self.create_collections_table()
                self.create_products_table()
                self.create_document_products_table()
                self.create_hdus_and_cards_tables()
            finally:
                self.conn.close()
                self.conn = None

    def populate_bundles_table(self):
        # type: () -> None
        self.conn.execute(BUNDLES_SQL, bundle_tuple(self.bundle))

    def populate_collections_table(self):
        # type: () -> None
        cs = [collection_tuple(c) for c in self.bundle.collections()]
        self.conn.executemany(COLLECTIONS_SQL, cs)

    def populate_products_table(self):
        # type: () -> None
        ps = [product_tuple(p)
              for c in self.bundle.collections()
              for p in c.products()]
        self.conn.executemany(PRODUCTS_SQL, ps)

    def populate_hdus_and_cards_tables(self):
        # type: () -> None
        def handle_undefined(val):
            """Convert undefined values to None"""
            if isinstance(val, pyfits.card.Undefined):
                return None
            else:
                return val

        def desired_keyword(kw):
            # type: (str) -> bool
            """Return True if the keyword is wanted"""
            # For now we want all of them.
            return kw is not None

        for p in self.bundle.products():
            try:
                fits = pyfits.open(p.absolute_filepath())
                try:
                    product_lid = str(p.lid)
                    self.conn.execute("""UPDATE products SET hdu_count = ?
                                         WHERE product=?""",
                                      (len(fits), product_lid))
                    for (hdu_index, hdu) in enumerate(fits):
                        fileinfo = hdu.fileinfo()
                        self.conn.execute(
                            HDUS_SQL,
                            (product_lid,
                             hdu_index,
                             fileinfo['hdrLoc'],
                             fileinfo['datLoc'],
                             fileinfo['datSpan']))
                        header = hdu.header
                        cs = [(card.keyword,
                               handle_undefined(card.value),
                               product_lid,
                               hdu_index)
                              for card in header.cards
                              if desired_keyword(card.keyword)]
                        self.conn.executemany(CARDS_SQL, cs)
                finally:
                    fits.close()
            except IOError as e:
                self.conn.execute(BAD_FITS_FILES_SQL, (str(p.lid), str(e)))

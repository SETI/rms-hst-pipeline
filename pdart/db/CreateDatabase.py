"""
This module creates a SQLite database from an archive.
"""
import os.path
import pyfits
import sqlite3

from pdart.db.TableSchemas import *
from pdart.pds4.Archives import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pds4.Archive


class DatabaseCreator(object):
    def __init__(self, conn, archive):
        # type: (sqlite3.Connection, pdart.pds4.Archive.Archive) -> None
        self.conn = conn
        self.archive = archive

    def create(self):
        """
        Given an open SQLite connection to a fresh database and an
        :class:`~pdart.pds4.Archive`, populate the database with the
        archive's information.
        """
        self.conn.execute('PRAGMA foreign_keys = ON;')
        self.create_bundles_table()
        self.create_collections_table()
        self.create_products_table()
        self.create_hdus_and_cards_tables()

    def create_bundles_table(self):
        # type: () -> None
        """Create the bundles table."""
        self.conn.execute('DROP TABLE IF EXISTS bundles')
        self.conn.execute(BUNDLES_SCHEMA)
        self.populate_bundles_table()
        self.conn.commit()

    def create_collections_table(self):
        # type: () -> None
        """Create the collections table."""
        self.conn.execute('DROP TABLE IF EXISTS collections')
        self.conn.execute(COLLECTIONS_SCHEMA)

        indexing = """CREATE INDEX idx_collections_bundle
                      ON collections(bundle)"""
        self.conn.execute(indexing)

        indexing = """CREATE INDEX idx_collections_prefix_suffix
                      ON collections(prefix, suffix)"""
        self.conn.execute(indexing)

        self.populate_collections_table()
        self.conn.commit()

    def create_products_table(self):
        # type: () -> None
        """Create the products table."""
        self.conn.execute('DROP TABLE IF EXISTS products')
        self.conn.execute(PRODUCTS_SCHEMA)

        indexing = """CREATE INDEX idx_products_collection
                      ON products(collection)"""
        self.conn.execute(indexing)

        self.populate_products_table()
        self.conn.commit()

    def create_hdus_and_cards_tables(self):
        # type: () -> None
        """Create the hdus and cards tables."""
        self.conn.execute('DROP TABLE IF EXISTS bad_fits_files')
        self.conn.execute(BAD_FITS_FILES_SCHEMA)

        self.conn.execute('DROP TABLE IF EXISTS hdus')
        self.conn.execute(HDUS_SCHEMA)

        indexing = """CREATE INDEX idx_hdus_product ON hdus(product)"""
        self.conn.execute(indexing)

        self.conn.execute('DROP TABLE IF EXISTS cards')
        self.conn.execute(CARDS_SCHEMA)

        indexing = """CREATE INDEX idx_cards_product_hdu_index
                      ON cards(product, hdu_index)"""
        self.conn.execute(indexing)
        self.populate_hdus_and_cards_tables()
        self.conn.commit()

    def populate_bundles_table(self):
        # type: () -> None
        bs = [(str(b.lid), b.absolute_filepath(),
               b.label_filepath(), b.proposal_id())
              for b in self.archive.bundles()]
        self.conn.executemany('INSERT INTO bundles VALUES (?,?,?,?)', bs)

    def populate_collections_table(self):
        # type: () -> None
        cs = [(str(c.lid), c.absolute_filepath(), c.label_filepath(),
               str(b.lid), c.prefix(), c.suffix(), c.instrument(),
               c.inventory_name(), c.inventory_filepath())
              for b in self.archive.bundles()
              for c in b.collections()]
        self.conn.executemany(
            'INSERT INTO collections VALUES (?,?,?,?,?,?,?,?,?)', cs)

    def populate_products_table(self):
        # type: () -> None
        ps = [(str(p.lid), p.absolute_filepath(),
               os.path.basename(p.absolute_filepath()),
               p.label_filepath(), str(c.lid),
               p.visit(), p.lid.product_id)
              for c in self.archive.collections()
              for p in c.products()]
        self.conn.executemany(
            'INSERT INTO products VALUES (?,?,?,?,?,?,0,?)', ps)

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

        for p in self.archive.products():
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
                            'INSERT INTO hdus VALUES (?, ?, ?, ?, ?)',
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
                        self.conn.executemany(
                            'INSERT INTO cards VALUES (?, ?, ?, ?)', cs)
                finally:
                    fits.close()
            except IOError as e:
                self.conn.execute('INSERT INTO bad_fits_files VALUES (?,?)',
                                  (str(p.lid), str(e)))

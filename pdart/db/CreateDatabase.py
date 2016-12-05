"""
This module creates a SQLite database from an archive.
"""
import os.path
import pyfits
import sqlite3

from pdart.pds4.Archives import *

import pdart.pds4.Archive  # for mypy


def _create_bundles_table(conn, archive):
    # type: (sqlite3.Connection, pdart.pds4.Archive.Archive) -> None
    """Create the bundles table."""
    conn.execute('DROP TABLE IF EXISTS bundles')
    table_creation = """CREATE TABLE bundles (
        bundle VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        proposal_id INT NOT NULL
        )"""
    conn.execute(table_creation)
    bs = [(str(b.lid), b.absolute_filepath(),
           b.label_filepath(), b.proposal_id())
          for b in archive.bundles()]
    conn.executemany('INSERT INTO bundles VALUES (?,?,?,?)', bs)
    conn.commit()


def _create_collections_table(conn, archive):
    # type: (sqlite3.Connection, pdart.pds4.Archive.Archive) -> None
    """Create the collections table."""
    conn.execute('DROP TABLE IF EXISTS collections')
    table_creation = """CREATE TABLE collections (
        collection VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        bundle VARCHAR NOT NULL,
        prefix VARCHAR NOT NULL,
        suffix VARCHAR NOT NULL,
        instrument VARCHAR NOT NULL,
        inventory_name VARCHAR NOT NULL,
        inventory_filepath VARCHAR NOT NULL,
        FOREIGN KEY(bundle) REFERENCES bundles(bundle)
            );"""
    conn.execute(table_creation)

    indexing = """CREATE INDEX idx_collections_bundle
                  ON collections(bundle)"""
    conn.execute(indexing)

    indexing = """CREATE INDEX idx_collections_prefix_suffix
                  ON collections(prefix, suffix)"""
    conn.execute(indexing)

    cs = [(str(c.lid), c.absolute_filepath(), c.label_filepath(),
           str(b.lid), c.prefix(), c.suffix(), c.instrument(),
           c.inventory_name(), c.inventory_filepath())
          for b in archive.bundles()
          for c in b.collections()]
    conn.executemany('INSERT INTO collections VALUES (?,?,?,?,?,?,?,?,?)', cs)
    conn.commit()


def _create_products_table(conn, archive):
    # type: (sqlite3.Connection, pdart.pds4.Archive.Archive) -> None
    """Create the products table."""
    conn.execute('DROP TABLE IF EXISTS products')
    table_creation = """CREATE TABLE products (
        product VARCHAR PRIMARY KEY NOT NULL,
        full_filepath VARCHAR NOT NULL,
        filename VARCHAR NOT NULL,
        label_filepath VARCHAR NOT NULL,
        collection VARCHAR NOT NULL,
        visit VARCHAR NOT NULL,
        hdu_count INT NOT NULL,
        product_id VARCHAR NOT NULL,
        FOREIGN KEY(collection) REFERENCES collections(collection)
        )"""
    conn.execute(table_creation)

    indexing = """CREATE INDEX idx_products_collection
                  ON products(collection)"""
    conn.execute(indexing)

    ps = [(str(p.lid), p.absolute_filepath(),
           os.path.basename(p.absolute_filepath()),
           p.label_filepath(), str(c.lid),
           p.visit(), p.lid.product_id)
          for c in archive.collections()
          for p in c.products()]
    conn.executemany('INSERT INTO products VALUES (?,?,?,?,?,?,0,?)', ps)
    conn.commit()


def _create_hdus_and_cards_tables(conn, archive):
    # type: (sqlite3.Connection, pdart.pds4.Archive.Archive) -> None
    """Create the hdus and cards tables."""
    def handle_undefined(val):
        """Convert undefined values to None"""
        if isinstance(val, pyfits.card.Undefined):
            return None
        else:
            return val

    def desired_keyword(kw):
        """Return True if the keyword is wanted"""
        return kw

    conn.execute('DROP TABLE IF EXISTS bad_fits_files')
    table_creation = """CREATE TABLE bad_fits_files (
        product VARCHAR NOT NULL,
        message VARCHAR NOT NULL,
        FOREIGN KEY (product) REFERENCES products(product)
        )"""
    conn.execute(table_creation)

    conn.execute('DROP TABLE IF EXISTS hdus')
    table_creation = """CREATE TABLE hdus (
        product VARCHAR NOT NULL,
        hdu_index INTEGER NOT NULL,
        hdrLoc INTEGER NOT NULL,
        datLoc INTEGER NOT NULL,
        datSpan INTEGER NOT NULL,
        FOREIGN KEY (product) REFERENCES products(product),
        CONSTRAINT hdus_pk PRIMARY KEY (product, hdu_index)
        )"""
    conn.execute(table_creation)

    indexing = """CREATE INDEX idx_hdus_product ON hdus(product)"""
    conn.execute(indexing)

    conn.execute('DROP TABLE IF EXISTS cards')
    table_creation = """CREATE TABLE cards (
        keyword VARCHAR NOT NULL,
        value,
        product VARCHAR NOT NULL,
        hdu_index INTEGER NOT NULL,
        FOREIGN KEY(product) REFERENCES products(product),
        FOREIGN KEY(product, hdu_index) REFERENCES hdus(product, hdu_index)
        )"""
    conn.execute(table_creation)

    indexing = """CREATE INDEX idx_cards_product_hdu_index
                  ON cards(product, hdu_index)"""
    conn.execute(indexing)

    for p in archive.products():
        try:
            fits = pyfits.open(p.absolute_filepath())
            try:
                product_lid = str(p.lid)
                conn.execute("""UPDATE products SET hdu_count = ?
                                WHERE product=?""",
                             (len(fits), product_lid))
                for (hdu_index, hdu) in enumerate(fits):
                    fileinfo = hdu.fileinfo()
                    conn.execute("""INSERT INTO hdus
                                    VALUES (?, ?, ?, ?, ?)""",
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
                    conn.executemany("""INSERT INTO cards
                                        VALUES (?, ?, ?, ?)""", cs)
            finally:
                fits.close()
        except IOError as e:
            conn.execute('INSERT INTO bad_fits_files VALUES (?,?)',
                         (str(p.lid), str(e)))

    conn.commit()


def create_database(conn, archive):
    # type: (sqlite3.Connection, pdart.pds4.Archive.Archive) -> None
    """
    Given an open SQLite connection to a fresh database and an
    :class:`~pdart.pds4.Archive`, populate the database with the
    archive's information.
    """
    conn.execute('PRAGMA foreign_keys = ON;')
    _create_bundles_table(conn, archive)
    _create_collections_table(conn, archive)
    _create_products_table(conn, archive)
    _create_hdus_and_cards_tables(conn, archive)

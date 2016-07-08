"""
SCRIPT: Run through the archive and extract information into a SQLite
database.
"""
import os.path
import pyfits
import sqlite3

from pdart.pds4.Archives import *


def makeBundlesTable(conn, archive):
    conn.execute('DROP TABLE IF EXISTS bundles')
    table_creation = """CREATE TABLE bundles (
        bundle VARCHAR PRIMARY KEY NOT NULL,
        label_filepath VARCHAR NOT NULL,
        proposal_id INT NOT NULL
        )"""
    conn.execute(table_creation)
    bs = [(str(b.lid), b.label_filepath(), b.proposal_id())
          for b in archive.bundles()]
    conn.executemany('INSERT INTO bundles VALUES (?, ?, ?)', bs)
    conn.commit()


def makeCollectionsTable(conn, archive):
    conn.execute('DROP TABLE IF EXISTS collections')
    table_creation = """CREATE TABLE collections (
        collection VARCHAR PRIMARY KEY NOT NULL,
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

    cs = [(str(c.lid), c.label_filepath(), str(b.lid),
           c.prefix(), c.suffix(), c.instrument(),
           c.inventory_name(), c.inventory_filepath())
          for b in archive.bundles()
          for c in b.collections()]
    conn.executemany('INSERT INTO collections VALUES (?,?,?,?,?,?,?,?)', cs)
    conn.commit()


def makeProductsTable(conn, archive):
    conn.execute('DROP TABLE IF EXISTS products')
    table_creation = """CREATE TABLE products (
        product VARCHAR PRIMARY KEY NOT NULL,
        label_filepath VARCHAR NOT NULL,
        collection VARCHAR NOT NULL,
        visit VARCHAR NOT NULL,
        FOREIGN KEY(collection) REFERENCES collections(collection)
        )"""
    conn.execute(table_creation)
    ps = [(str(p.lid), p.label_filepath(), str(c.lid), p.visit())
          for c in archive.collections()
          for p in c.products()]
    conn.executemany('INSERT INTO products VALUES (?, ?, ?, ?)', ps)
    conn.commit()


def makeHdusAndCardsTables(conn, archive):
    def handle_undefined(val):
        """Convert undefined values to None"""
        if isinstance(val, pyfits.card.Undefined):
            return None
        else:
            return val

    def desired_keyword(kw):
        """Return True if the keyword is wanted"""
        return kw

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

    for p in archive.products():
        try:
            fits = pyfits.open(p.absolute_filepath())
            try:
                product_lid = str(p.lid)
                for (hdu_index, hdu) in enumerate(fits):
                    fileinfo = hdu.fileinfo()
                    conn.execute('INSERT INTO hdus ' +
                                 'VALUES (?, ?, ?, ?, ?)',
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
                    conn.executemany('INSERT INTO CARDS ' +
                                     'VALUES (?, ?, ?, ?)', cs)
            finally:
                fits.close()
        except IOError:
            pass

    conn.commit()


if __name__ == '__main__':
    archive = get_any_archive()
    archive_dir = archive.root
    db_filepath = os.path.join(archive_dir, 'archive.spike.db')
    conn = sqlite3.connect(db_filepath)
    conn.execute('PRAGMA foreign_keys = ON;')
    try:
        makeBundlesTable(conn, archive)
        makeCollectionsTable(conn, archive)
        makeProductsTable(conn, archive)
        makeHdusAndCardsTables(conn, archive)
    finally:
        conn.close()

    # dump it
    dump = False
    if dump:
        conn = sqlite3.connect(db_filepath)
        for line in conn.iterdump():
            print line
        conn.close()

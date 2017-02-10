"""
TODO Document this and add tests.
"""
from contextlib import closing
import os.path
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.pds4.Bundle import Bundle
    from pdart.pds4.LID import LID


##############################
# Database stuff
##############################

_NEW_DATABASE_NAME = 'complete-database.db'
# type: str


def bundle_database_filepath(bundle):
    # type: (Bundle) -> unicode
    return os.path.join(bundle.absolute_filepath(), _NEW_DATABASE_NAME)


def open_bundle_database(bundle):
    # type: (Bundle) -> sqlite3.Connection
    return sqlite3.connect(bundle_database_filepath(bundle))


def init_database(conn):
    # type: (sqlite3.Connection) -> None
    with closing(conn.cursor()) as cursor:
        cursor.execute('PRAGMA foreign_keys=ON;')


##############################
# FITS products stuff
##############################

def _exists_database_record_for_fits_in_table(cursor, lid, table_name):
    # type: (sqlite3.Cursor, LID, str) -> bool
    VERBOSE = False
    try:
        cursor.execute('SELECT count(lid) FROM %s WHERE lid=?' % table_name,
                       (str(lid),))
        (res,) = cursor.fetchone()
        if res is 0:
            if VERBOSE:
                print 'Failed for', lid, table_name
            return False
        return True
    except Exception as e:
        if VERBOSE:
            print 'Failed:', e
        return False


def exists_database_records_for_fits(conn, lid):
    # type: (sqlite3.Connection, LID) -> bool
    VERBOSE = False
    with closing(conn.cursor()) as cursor:
        try:
            f = _exists_database_record_for_fits_in_table
            # TODO I should also be checking that product_type in the
            # products table = 'fits'.
            return f(cursor, lid, 'products') and \
                f(cursor, lid, 'fits_products')

        except Exception as e:
            if VERBOSE:
                print 'threw', str(lid), e
            return False


def ensure_collections_table(cursor):
    # type: (sqlite3.Cursor) -> None
    cursor.execute("""CREATE TABLE IF NOT EXISTS collections (
                      lid VARCHAR PRIMARY KEY NOT NULL);""")


def ensure_products_table(cursor):
    # type: (sqlite3.Cursor) -> None
    cursor.execute("""CREATE TABLE IF NOT EXISTS products (
                      lid VARCHAR PRIMARY KEY NOT NULL,
                      collection VARCHAR NOT NULL,
                      product_type VARCHAR NOT NULL,
                      FOREIGN KEY(collection) REFERENCES collections(lid));""")


def ensure_fits_products_table(cursor):
    # type: (sqlite3.Cursor) -> None
    cursor.execute("""CREATE TABLE IF NOT EXISTS fits_products (
                      lid VARCHAR PRIMARY KEY NOT NULL,
                      FOREIGN KEY(lid) REFERENCES products(lid));""")


def insert_fits_database_records(cursor, lid_obj):
    # type: (sqlite3.Cursor, LID) -> None

    lid = str(lid_obj)
    collection_lid = str(lid_obj.parent_lid())
    # TODO could combine these
    ensure_collections_table(cursor)
    ensure_products_table(cursor)
    ensure_fits_products_table(cursor)
    cursor.execute("INSERT INTO products VALUES(?,?,'fits');",
                   (lid, collection_lid))
    cursor.execute('INSERT INTO fits_products VALUES(?);', (lid,))


##############################
# Browse products stuff
##############################

def exists_database_records_for_browse(conn, lid):
    # type: (sqlite3.Connection, LID) -> bool
    VERBOSE = False
    with closing(conn.cursor()) as cursor:
        try:
            cursor.execute('SELECT count(lid) FROM products WHERE lid=?',
                           (str(lid),))
            (res,) = cursor.fetchone()
            if res is 0:
                if VERBOSE:
                    print 'Failed for', lid
                return False
            return True

        except Exception as e:
            if VERBOSE:
                print 'threw', str(lid), e
            return False


def insert_browse_database_records(cursor, lid_obj):
    # type: (sqlite3.Cursor, LID) -> None

    lid = str(lid_obj)
    collection_lid = str(lid_obj.parent_lid())
    # TODO could combine these
    ensure_products_table(cursor)
    cursor.execute("INSERT INTO products VALUES(?,?,'browse');",
                   (lid, collection_lid))

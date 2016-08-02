"""
SCRIPT: Build the database then build bundle, collection, and product
labels, and collection inventories.  This is a temporary script for
development.
"""
from contextlib import closing
import sqlite3

from MakeDB import *
from pdart.exceptions.Combinators import *
from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *

VERIFY = False
IN_MEMORY = False


def make_db_bundle_labels(conn):
    with closing(conn.cursor()) as cursor:
        for (lid,) in cursor.execute('SELECT bundle FROM bundles'):
            assert isinstance(lid, unicode), type(lid)
            make_db_bundle_label(conn, lid, VERIFY)


def make_db_collection_labels_and_inventories(conn):
    with closing(conn.cursor()) as cursor:
        for (lid,) in cursor.execute('SELECT collection FROM collections'):
            make_db_collection_label_and_inventory(conn, lid, VERIFY)


def make_db_product_labels(conn):
    with closing(conn.cursor()) as cursor:
        for (lid,) in cursor.execute(
            """SELECT product FROM products EXCEPT
               SELECT product FROM bad_fits_files"""):
            assert isinstance(lid, unicode), type(lid)
            make_db_product_label(conn, lid, VERIFY)


def getConn():
    if IN_MEMORY:
        return sqlite3.connect(':memory:')
    else:
        return sqlite3.connect(
            '/Users/spaceman/Desktop/Archive/archive.spike.db')


def dev():
    archive = get_any_archive()
    with closing(getConn()) as conn:
        makeDB(conn, archive)
        make_db_product_labels(conn)
        make_db_collection_labels_and_inventories(conn)
        make_db_bundle_labels(conn)


if __name__ == '__main__':
    raise_verbosely(dev)

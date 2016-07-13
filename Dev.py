"""
SCRIPT: Run through the database and build bundle labels, collection
labels, and collection inventories.  This is a temporary script for
development.
"""
from contextlib import closing
import sqlite3

from pdart.exceptions.Combinators import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *

VERIFY = False


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
        for (lid,) in cursor.execute('SELECT product FROM products'):
            assert isinstance(lid, unicode), type(lid)
            with closing(conn.cursor()) as cursor2:
                cursor2.execute(
                    'SELECT message FROM bad_fits_files WHERE product=?',
                    (str(lid),))
                res = cursor2.fetchone()
            if res is None:
                make_db_product_label(conn, lid, VERIFY)
            else:
                print 'bad fits file for', str(lid)


def dev():
    with closing(sqlite3.connect(
            '/Users/spaceman/Desktop/Archive/archive.spike.db')) as conn:
        make_db_collection_labels_and_inventories(conn)
        make_db_product_labels(conn)
        make_db_bundle_labels(conn)

if __name__ == '__main__':
    raise_verbosely(dev)

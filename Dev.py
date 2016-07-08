"""
SCRIPT: Run through the database and build bundle labels, collection
labels, and collection inventories.  This is a temporary script for
development.
"""
import sqlite3

from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *


def make_db_bundle_labels(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT bundle FROM bundles')
    bundle_lids = [lid for (lid,) in cursor.fetchall()]
    for lid in bundle_lids:
        assert isinstance(lid, unicode), type(lid)
        make_db_bundle_label(conn, lid, True)


def make_db_collection_labels_and_inventories(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT collection FROM collections')
    collection_lids = [lid for (lid,) in cursor.fetchall()]
    for lid in collection_lids:
        make_db_collection_label_and_inventory(conn, lid, True)

if __name__ == '__main__':
    conn = sqlite3.connect('/Users/spaceman/Desktop/Archive/archive.spike.db')
    try:
        make_db_collection_labels_and_inventories(conn)
        make_db_bundle_labels(conn)
    finally:
        conn.close()

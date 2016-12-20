"""
**SCRIPT:** Build the database then build bundle, collection, and
product labels, and collection inventories.  This is a temporary
script under active development and so is underdocumented.
"""
from contextlib import closing
import os.path
import sqlite3

from pdart.db.CreateDatabase import ArchiveDatabaseCreator
from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.ProductLabel import *
from pdart.rules.Combinators import *

from typing import cast, Iterable


VERIFY = False
# type: bool
IN_MEMORY = False
# type: bool
CREATE_DB = True
# type: bool


def make_db_labels(conn):
    # type: (sqlite3.Connection) -> None
    """
    Write labels for the whole archive in hierarchical order to
    increase cache hits for bundles and collections.  NOTE: This
    doesn't seem to run any faster, perhaps because of extra cost to
    having three open cursors.
    """
    with closing(conn.cursor()) as bundle_cursor:
        for (bundle,) in cast(Iterable[Tuple[unicode]],
                              bundle_cursor.execute(
                'SELECT bundle FROM bundles')):

            with closing(conn.cursor()) as collection_cursor:
                for (coll,) in cast(Iterable[Tuple[unicode]],
                                    collection_cursor.execute(
                  'SELECT collection FROM collections WHERE bundle=?',
                  (bundle,))):

                    with closing(conn.cursor()) as product_cursor:
                        for (prod,) in cast(Iterable[Tuple[unicode]],
                                            product_cursor.execute(
                                """SELECT product FROM products WHERE collection=?
                           EXCEPT SELECT product FROM bad_fits_files""",
                                (coll,))):

                            make_db_product_label(conn, prod, VERIFY)

                    make_db_collection_label_and_inventory(conn, coll, VERIFY)

            make_db_bundle_label(conn, bundle, VERIFY)


def make_db_bundle_labels(conn):
    # type: (sqlite3.Connection) -> None
    with closing(conn.cursor()) as cursor:
        for (lid,) in cast(Iterable[Tuple[unicode]],
                           cursor.execute('SELECT bundle FROM bundles')):
            make_db_bundle_label(conn, lid, VERIFY)


def make_db_collection_labels_and_inventories(conn):
    # type: (sqlite3.Connection) -> None
    with closing(conn.cursor()) as cursor:
        for (lid,) in cast(Iterable[Tuple[unicode]],
                           cursor.execute(
                'SELECT collection FROM collections')):
            make_db_collection_label_and_inventory(conn, lid, VERIFY)


def make_db_product_labels(conn):
    # type: (sqlite3.Connection) -> None
    with closing(conn.cursor()) as cursor:
        for (lid,) in cast(Iterable[Tuple[unicode]], cursor.execute(
            """SELECT product FROM products EXCEPT
               SELECT product FROM bad_fits_files""")):
            make_db_product_label(conn, lid, VERIFY)


def get_conn():
    # type: () -> sqlite3.Connection
    if IN_MEMORY:
        return sqlite3.connect(':memory:')
    else:
        return sqlite3.connect(os.path.join(get_any_archive_dir(),
                                            DATABASE_NAME))


def dev():
    # type: () -> None
    archive = get_any_archive()
    with closing(get_conn()) as conn:
        if CREATE_DB:
            ArchiveDatabaseCreator(conn, archive).create()
        # It seems to run about the same, building labels
        # hierarchically or by type.
        if True:
            make_db_labels(conn)
        else:
            make_db_product_labels(conn)
            make_db_collection_labels_and_inventories(conn)
            make_db_bundle_labels(conn)


if __name__ == '__main__':
    raise_verbosely(dev)

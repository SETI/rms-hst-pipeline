"""
**SCRIPT:** Build the bundle databases then build bundle, collection,
and product labels, and collection inventories.  Uses the bundle
databases.
"""
from contextlib import closing
import os.path
import sqlite3

from pdart.db.CreateBundleDatabase import BundleDatabaseCreator
from pdart.db.DatabaseName import DATABASE_NAME
from pdart.pds4.Archive import Archive
from pdart.pds4.Archives import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.ProductLabel import *
from pdart.rules.Combinators import *

from typing import cast, Iterable


VERIFY = False
# type: bool
IN_MEMORY = False
# type: bool
CREATE_DB = True
# type: bool


def make_db_labels(archive):
    # type: (Archive) -> None
    """
    Write labels for the whole archive in hierarchical order to
    increase cache hits for bundles and collections.  NOTE: This
    doesn't seem to run any faster, perhaps because of extra cost to
    having three open cursors.
    """
    for bundle_obj in archive.bundles():
        bundle = bundle_obj.lid.lid
        database_fp = os.path.join(bundle_obj.absolute_filepath(),
                                   DATABASE_NAME)
        with closing(sqlite3.connect(database_fp)) as conn:
            with closing(conn.cursor()) as collection_cursor:
                for (coll,) in get_bundle_collections_db(collection_cursor,
                                                         bundle):

                    with closing(conn.cursor()) as product_cursor:
                        prod_iter = get_good_collection_products_db(
                            product_cursor, coll)
                        for (prod,) in prod_iter:

                            make_db_product_label(conn, prod, VERIFY)

                    make_db_collection_label_and_inventory(conn, coll, VERIFY)

            make_db_bundle_label(conn, bundle, VERIFY)


def dev():
    # type: () -> None
    archive = get_any_archive()

    if CREATE_DB:
        BundleDatabaseCreator(archive).create()

    make_db_labels(archive)


if __name__ == '__main__':
    raise_verbosely(dev)

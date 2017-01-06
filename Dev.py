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

from typing import cast, Iterable, TYPE_CHECKING

VERIFY = False
# type: bool
IN_MEMORY = False
# type: bool
CREATE_DB = True
# type: bool


class ArchiveRecursion(object):
    def run(self, archive):
        # type: (Archive) -> None
        for bundle in archive.bundles():
            database_fp = os.path.join(bundle.absolute_filepath(),
                                       DATABASE_NAME)
            with closing(sqlite3.connect(database_fp)) as conn:
                with closing(conn.cursor()) as collection_cursor:
                    for (coll,) in get_bundle_collections_db(collection_cursor,
                                                             bundle.lid.lid):
                        with closing(conn.cursor()) as product_cursor:
                            prod_iter = get_good_collection_products_db(
                                product_cursor, coll)
                            for (prod,) in prod_iter:
                                self.handle_product(archive, conn, prod)
                        self.handle_collection(archive, conn, coll)
                self.handle_bundle(archive, conn, bundle.lid.lid)

    def handle_bundle(self, archive, conn, bundle_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> int
        pass

    def handle_collection(self, archive, conn, collection_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> int
        pass

    def handle_product(self, archive, conn, product_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> int
        pass


class LabelCreationRecursion(ArchiveRecursion):
    def handle_bundle(self, archive, conn, bundle_lid):
        make_db_bundle_label(conn, bundle_lid, VERIFY)

    def handle_collection(self, archive, conn, collection_lid):
        make_db_collection_label_and_inventory(conn, collection_lid, VERIFY)

    def handle_product(self, archive, conn, product_lid):
        make_db_product_label(conn, product_lid, VERIFY)


def dev():
    # type: () -> None
    archive = get_any_archive()

    if CREATE_DB:
        BundleDatabaseCreator(archive).create()

    LabelCreationRecursion().run(archive)


if __name__ == '__main__':
    raise_verbosely(dev)

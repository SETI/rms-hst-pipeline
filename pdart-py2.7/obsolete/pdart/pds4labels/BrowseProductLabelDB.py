"""
Functionality to build a raw browse product label using a SQLite
database.
"""
from contextlib import closing
from os.path import getsize
import sys

from pdart.pds4.LID import *
from pdart.pds4.Product import *
from pdart.pds4labels.BrowseProductLabelXml import *
from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.xml.Pretty import *

from typing import Iterable, Tuple
if TYPE_CHECKING:
    import sqlite3
    from typing import Iterable, Tuple
    from pdart.pds4.Archive import Archive


def make_db_browse_product_labels(archive, conn):
    # type: (Archive, sqlite3.Connection) -> None
    """
    Given an :class:`~pdart.pds4.Archive` and a database connection,
    create PDS4 labels for the browse products.
    """

    # TODO Polish (with SQL joins?).  First, an inefficient
    # implementation.
    with closing(conn.cursor()) as cursor:
        colls = [cfbs for suff in RAW_SUFFIXES
                 for cfbs in get_data_collections_info_db(cursor, suff)]

        for (c, _, b, suffix) in colls:
            (_, proposal_id) = get_bundle_info_db(cursor, b)
            # print '>>>>', c
            for (p,) in get_good_collection_products_db(cursor, c):
                lid = LID(p)
                product = Product(archive, lid)
                browse_product = product.browse_product()
                browse_file_name = lid.product_id + '.jpg'
                browse_image_file = list(browse_product.files())[0]
                object_length = getsize(
                    browse_image_file.full_filepath())
                label = make_label({
                        'proposal_id': str(proposal_id),
                        'suffix': suffix,
                        'browse_lid': str(browse_product.lid),
                        'data_lid': str(lid),
                        'browse_file_name': browse_file_name,
                        'object_length': str(object_length)
                        }).toxml()

                label_fp = browse_product.label_filepath()

                with open(label_fp, 'w') as f:
                    f.write(label)

                print 'browse product label for', browse_product.lid
                sys.stdout.flush()


def make_db_collection_browse_product_labels(archive, conn, collection):
    # type: (Archive, sqlite3.Connection, unicode) -> None
    """
    Given an :class:`~pdart.pds4.Archive`, a bundle's database
    connection, and a collection LID, create PDS4 labels for the
    browse products in that collection.
    """

    # TODO Polish (with SQL joins?).  First, an inefficient
    # implementation.
    with closing(conn.cursor()) as cursor:
        colls = [fbs for suff in RAW_SUFFIXES
                 for fbs in get_data_collection_info_by_suffix_db(cursor,
                                                                  collection,
                                                                  suff)]

        for (_, b, suffix) in colls:
            (_, proposal_id) = get_bundle_info_db(cursor, b)
            # print '>>>>', c
            for (p,) in get_good_collection_products_db(cursor, collection):
                lid = LID(p)
                product = Product(archive, lid)
                browse_product = product.browse_product()
                browse_file_name = lid.product_id + '.jpg'
                browse_image_file = list(browse_product.files())[0]
                object_length = getsize(
                    browse_image_file.full_filepath())
                label = make_label({
                        'proposal_id': str(proposal_id),
                        'suffix': suffix,
                        'browse_lid': str(browse_product.lid),
                        'data_lid': str(lid),
                        'browse_file_name': browse_file_name,
                        'object_length': str(object_length)
                        }).toxml()

                label_fp = browse_product.label_filepath()

                with open(label_fp, 'w') as f:
                    f.write(label)

                print 'browse product label for', browse_product.lid
                sys.stdout.flush()
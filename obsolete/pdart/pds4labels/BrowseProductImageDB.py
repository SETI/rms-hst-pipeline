"""
Functionality to build a RAW browse product image using a SQLite
database.
"""
from contextlib import closing
import sys

from fs.path import basename, join, splitext

from pdart.db.TableSchemas import *
from pdart.pds4.LID import *
from pdart.pds4.Product import *
from pdart.pds4labels.BrowseProductImageReduction import ensure_directory
from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES

import pdart.add_pds_tools
import picmaker

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3
    from typing import Iterable, Tuple
    from pdart.pds4.Archive import Archive


def _make_browse_coll_fp(raw_coll_fp):
    # type: (str) -> str
    """
    Given the filepath to the RAW collection, return the filepath to
    its browse collection.
    """
    # TODO Sloppy implementation: assumes only one 'data'
    res = 'browse'.join(re.split('data', raw_coll_fp))
    ensure_directory(res)
    return res


def _make_browse_image(browse_coll_fp, raw_full_filepath, visit):
    # type: (str, str, unicode) -> None
    """
    Given a filepath for the destination browse collection, a filepath
    for the source RAW file, and the product's visit code, create the
    browse image and save it.
    """
    basename = basename(raw_full_filepath)
    basename = splitext(basename)[0] + '.jpg'
    target_dir = join(browse_coll_fp, ('visit_%s' % visit))

    ensure_directory(target_dir)

    # TODO: WFPC2 gives four images, not one, so we'll have to do
    # something different here.  This probably just builds the first
    # image.  Add keyword wfpc2=True.
    picmaker.ImagesToPics([str(raw_full_filepath)],
                          str(target_dir),
                          filter='None',
                          percentiles=(0.1, 99.9))


def make_db_browse_product_images(archive, conn):
    # type: (Archive, sqlite3.Connection) -> None
    """
    Given a connection to a bundle's database, create browse images
    for all the RAW products in the bundle.
    """

    # TODO Polish (with SQL joins?).  First, an inefficient
    # implementation.
    with closing(conn.cursor()) as cursor:
        colls = [ccbs for suff in RAW_SUFFIXES
                 for ccbs in get_data_collections_info_db(cursor, suff)]

        for (c, c_fp, b, suffix) in colls:
            (_, proposal_id) = get_bundle_info_db(cursor, b)
            for (p, fp, v) \
                    in get_good_collection_products_with_info_db(cursor, c):
                print 'browse product images for', \
                    Product(archive, LID(p)).browse_product().lid
                sys.stdout.flush()

                browse_coll_fp = _make_browse_coll_fp(c_fp)
                _make_browse_image(browse_coll_fp, fp, v)


def make_db_collection_browse_product_images(archive, conn, collection):
    # type: (Archive, sqlite3.Connection, unicode) -> None
    """
    Given a connection to a bundle's database and a collection LID,
    create browse images for all the RAW products in that collection.
    """

    # TODO Polish (with SQL joins?).  First, an inefficient
    # implementation.
    with closing(conn.cursor()) as cursor:
        colls = [cbs for suff in RAW_SUFFIXES
                 for cbs
                 in get_data_collection_info_by_suffix_db(cursor,
                                                          collection,
                                                          suff)]

        for (c_fp, b, suffix) in colls:
            (_, proposal_id) = get_bundle_info_db(cursor, b)
            for (p, fp, v) \
                    in get_good_collection_products_with_info_db(cursor,
                                                                 collection):
                browse_prod = Product(archive, LID(p)).browse_product()
                print 'browse product images for', browse_prod.lid
                sys.stdout.flush()
                browse_coll_fp = _make_browse_coll_fp(c_fp)
                _make_browse_image(browse_coll_fp, fp, v)
                with closing(conn.cursor()) as cursor2:
                    add_product(cursor2, browse_prod)


def add_product(cursor, product):
    # type: (sqlite3.Cursor, Product) -> None
    cursor.execute(PRODUCTS_SQL, product_tuple(product))

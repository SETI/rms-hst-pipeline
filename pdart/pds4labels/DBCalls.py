from contextlib import closing
import sqlite3

from fs.path import join

from pdart.db.DatabaseName import DATABASE_NAME

from typing import cast, Iterable, Tuple, TYPE_CHECKING
# shorthand types for casting
_Tuple3 = Tuple[str, unicode, unicode]
_Tuple4 = Tuple[unicode, str, unicode, unicode]
_Tuple5 = Tuple[unicode, unicode, unicode, unicode, int]
_Tuple7 = Tuple[unicode, unicode, str, unicode, str, unicode, unicode]

if TYPE_CHECKING:
    from typing import Any, Dict, List

    from pdart.pds4.Archive import *
    from pdart.pds4.Bundle import *

    # shorthand types for type signatures
    HeaderDict = Dict[str, Any]
    Headers = List[HeaderDict]


def archive_database_filepath(archive):
    # type: (Archive) -> unicode
    return join(archive.root, DATABASE_NAME)


def open_archive_database(archive):
    # type: (Archive) -> sqlite3.Connection
    return sqlite3.connect(archive_database_filepath(archive))


def bundle_database_filepath(bundle):
    # type: (Bundle) -> unicode
    return join(bundle.absolute_filepath(), DATABASE_NAME)


def open_bundle_database(bundle):
    # type: (Bundle) -> sqlite3.Connection
    return sqlite3.connect(bundle_database_filepath(bundle))


def database_table_exists(conn, table):
    # type: (sqlite3.Connection, str) -> bool
    with closing(conn.cursor()) as cursor:
        try:
            cursor.execute('SELECT count(*) FROM %s' % table)
            return True
        except sqlite3.OperationalError:
            return False


def database_is_initialized(conn):
    # type: (sqlite3.Connection) -> bool
    all_tables = ['bundles', 'collections', 'products', 'bad_fits_files',
                  'hdus', 'cards', 'document_products', 'documents']
    res = True
    for table in all_tables:
        if not database_table_exists(conn, table):
            res = False
    return res


def get_all_bundles(cursor):
    # type: (sqlite3.Cursor) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute('SELECT bundle FROM bundles'))


def get_all_collections(cursor):
    # type: (sqlite3.Cursor) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute('SELECT collection FROM collections'))


def get_all_browse_collections(cursor):
    # type: (sqlite3.Cursor) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute(
            "SELECT collection FROM collections WHERE prefix='browse'"))


def get_all_browse_products(cursor):
    # type: (sqlite3.Cursor) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute(
            """SELECT product FROM products WHERE collection IN
               (SELECT collection FROM collections WHERE prefix='browse')"""))


def get_all_good_bundle_products(cursor, bundle):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute("""SELECT product FROM products
                                  WHERE collection IN
                                  (SELECT collection from collections
                                   WHERE bundle=?) EXCEPT
                                  SELECT product FROM bad_fits_files""",
                               (bundle,)))


def get_all_good_collection_products(cursor, collection):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute("""SELECT product FROM products
                                  WHERE collection=? EXCEPT
                                  SELECT product FROM bad_fits_files""",
                               (collection,)))


def get_bundle_info_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Tuple[str, int]
    """Returns (label_filepath, proposal_id)"""
    cursor.execute("""SELECT label_filepath, proposal_id
                      FROM bundles WHERE bundle=?""",
                   (lid,))
    return cast(Tuple[str, int], cursor.fetchone())


def get_bundle_collections_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute(
            'SELECT collection from collections WHERE bundle=?', (lid,)))


def get_collection_info_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> _Tuple7
    """Returns (bundle, instrument, inventory_filepath, inventory_name,
    label_filepath, prefix, suffix)"""

    cursor.execute("""SELECT bundle, instrument, inventory_filepath,
                      inventory_name, label_filepath, prefix, suffix
                      FROM collections WHERE collection=?""",
                   (lid,))
    return cast(_Tuple7, cursor.fetchone())


def get_data_collection_info_by_suffix_db(cursor, collection_lid, suffix):
    # type: (sqlite3.Cursor, unicode, unicode) -> Iterable[_Tuple3]
    """Given a collection LID and a suffix, returns the (collection,
    full_filepath, bundle, suffix) tuple for any data collection with
    that suffix"""
    return cast(Iterable[_Tuple3],
                cursor.execute(
            """SELECT full_filepath, bundle, suffix
               FROM collections
               WHERE collection=? AND prefix='data' AND suffix=?""",
            (collection_lid, suffix)))


def get_data_collections_info_db(cursor, suffix):
    # type: (sqlite3.Cursor, unicode) -> Iterable[_Tuple4]
    """Given a suffix, returns a (collection, full_filepath, bundle,
    suffix) tuple for any data collection with that suffix"""
    return cast(Iterable[_Tuple4],
                cursor.execute(
            """SELECT collection, full_filepath, bundle, suffix
               FROM collections WHERE prefix='data' AND suffix=?""",
            (suffix,)))


def get_collection_products_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute(
            'SELECT product FROM products WHERE collection=?',
            (lid,)))


def get_good_collection_products_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode]]
    return cast(Iterable[Tuple[unicode]],
                cursor.execute(
            """SELECT product FROM products WHERE collection=?
               EXCEPT SELECT product FROM bad_fits_files""",
            (lid,)))


def get_good_collection_products_with_info_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[unicode, str, unicode]]
    return cast(Iterable[Tuple[unicode, str, unicode]],
                cursor.execute(
            """SELECT product, full_filepath, visit FROM products
               WHERE collection=?
               AND product NOT IN (SELECT product FROM bad_fits_files)""",
            (lid,)))


def get_product_info_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> _Tuple5
    """Returns (file_name, label_fp, collection, product_id, hdu_count)"""
    cursor.execute(
        """SELECT filename, label_filepath, collection,
                  product_id, hdu_count
           FROM products WHERE product=?""",
        (lid,))
    return cast(_Tuple5, cursor.fetchone())


def get_fits_file_offsets_db(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Iterable[Tuple[int, int, int, int]]
    """Returns an Iterable of tuples (hdu_index, hdrLoc, datLoc, datSpan)"""
    return cast(Iterable[Tuple[int, int, int, int]],
                cursor.execute(
            """SELECT hdu_index, hdrLoc, datLoc, datSpan
               FROM hdus WHERE product=?
               ORDER BY hdu_index ASC""",
            (lid,)))


def get_fits_headers_db(cursor, lid, hdu_index):
    # type: (sqlite3.Cursor, unicode, int) -> HeaderDict
    """Returns an Iterable of pairs of keywords and values."""

    # We know that since it's coming from FITS headers, they are pairs
    # of strings and Anys.
    iter = cast(Iterable[Tuple[str, Any]],
                cursor.execute("""SELECT keyword, value FROM cards
                                  WHERE product=? AND hdu_index=?""",
                               (lid, hdu_index)))
    return dict(iter)


def get_document_product_info(cursor, lid):
    # type: (sqlite3.Cursor, unicode) -> Tuple[unicode, int]
    """Returns (label_filepath, proposal_id)"""
    cursor.execute("""SELECT label_filepath, proposal_id FROM document_products
                      WHERE product=?""", (lid,))

    return cast(Tuple[unicode, int], cursor.fetchone())


def delete_browse_products_and_collections(cursor):
    # type: (sqlite3.Cursor) -> None
    cursor.execute("""DELETE FROM products WHERE collection IN
                      (SELECT collection FROM collections
                       WHERE prefix='browse')""")
    cursor.execute("""DELETE FROM collections WHERE prefix='browse'""")

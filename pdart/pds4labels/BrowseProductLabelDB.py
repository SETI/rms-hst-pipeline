"""
Functionality to build a raw browse product label using a SQLite
database.
"""
from contextlib import closing

from pdart.pds4.LID import *
from pdart.pds4.Product import *
from pdart.pds4labels.BrowseProductLabelXml import *
from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.xml.Pretty import *

from typing import cast, Iterable, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.pds4.Archive import Archive
    import sqlite3


def make_db_browse_product_labels(conn, archive):
    # type: (sqlite3.Connection, Archive) -> None
    """
    Given a database connection and an :class:`~pdart.pds4.Archive`,
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
                object_length = os.path.getsize(
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

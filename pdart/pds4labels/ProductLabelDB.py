"""
Functionality to build a product label using a SQLite database.
"""
from contextlib import closing

from pdart.pds4labels.DatabaseCaches import *
from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.FileContentsDB import *
from pdart.pds4labels.HstParametersDB import *
from pdart.pds4labels.ObservingSystem import *
from pdart.pds4labels.TargetIdentificationDB import *
from pdart.pds4labels.TimeCoordinatesDB import *
from pdart.pds4labels.ProductLabelXml import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3
    from typing import Any, Dict, Iterable
    from pdart.pds4labels.DBCalls import Headers


def make_db_product_label(conn, lid, verify):
    # type: (sqlite3.Connection, unicode, bool) -> unicode
    """
    Create the label text for the product having this
    :class:`~pdart.pds4.LID` using the database connection.  If verify
    is True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    with closing(conn.cursor()) as cursor:
        (file_name, label_fp, collection,
         product_id, hdu_count) = get_product_info_db(cursor, lid)

        d = lookup_collection(conn, collection)
        bundle = d['bundle']
        instrument = d['instrument']
        suffix = d['suffix']

        d = lookup_bundle(conn, bundle)
        proposal_id = d['proposal_id']

        headers = _make_header_dictionaries(lid, hdu_count, cursor)

    label = make_label({
            'lid': str(lid),
            'proposal_id': str(proposal_id),
            'suffix': suffix,
            'file_name': file_name,
            'file_contents': get_db_file_contents(headers, conn, lid),
            'Investigation_Area_name': mk_Investigation_Area_name(proposal_id),
            'investigation_lidvid': mk_Investigation_Area_lidvid(proposal_id),
            'Observing_System': observing_system(instrument),
            'Time_Coordinates': get_db_time_coordinates(headers),
            'Target_Identification': get_db_target(headers),
            'HST': get_db_hst_parameters(headers, instrument, product_id)
            }).toxml()
    label = pretty_print(label)

    with open(label_fp, 'w') as f:
        f.write(label)

    print 'product label for', lid
    sys.stdout.flush()

    if verify:
        verify_label_or_raise(label)

    return label


def _make_header_dictionaries(lid, hdu_count, cursor):
    # type: (unicode, int, sqlite3.Cursor) -> Headers
    return [get_fits_headers_db(cursor, lid, i) for i in range(hdu_count)]

from contextlib import closing
from datetime import date
import sys

from pdart.pds4labels.DBCalls import get_document_product_info
from pdart.pds4labels.DocumentProductLabelXml import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3


def make_db_document_product_label(conn, bundle_lid, verify):
    # type: (sqlite3.Connection, unicode, bool) -> unicode
    """
    Create the label text for the document product in the bundle
    having this :class:`~pdart.pds4.LID` using the database
    connection.  If verify is True, verify the label against its XML
    and Schematron schemas.  Raise an exception if either fails.
    """

    product_lid = bundle_lid + ':document:phase2'
    with closing(conn.cursor()) as cursor:
        # get some unknown info
        (label_fp, proposal_id) = get_document_product_info(cursor,
                                                            product_lid)
        pass

    title = 'Summary of the observation plan for HST proposal %d' % proposal_id

    label = make_label({
            'bundle_lid': bundle_lid,
            'product_lid': product_lid,
            'title': title,
            'publication_date': date.today().isoformat(),
            'Citation_Information': make_citation_information(bundle_lid,
                                                              proposal_id),
            'Document_Edition': make_document_edition(
                '0.0',
                [('phase2.txt', '7-Bit ASCII Text')])
            }).toxml()
    label = pretty_print(label)

    with open(label_fp, 'w') as f:
        f.write(label)

    print 'product label for', product_lid
    sys.stdout.flush()

    if verify:
        verify_label_or_raise(label)

    return label

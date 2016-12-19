"""
Functionality to build a collection label using a SQLite database.
"""
from contextlib import closing
import io
import sys

from pdart.pds4labels.CollectionLabelXml import *
from pdart.pds4labels.DatabaseCaches import *
from pdart.xml.Pretty import *
from pdart.xml.Schema import *

from typing import cast, Iterable


def make_db_collection_inventory(conn, collection_lid):
    # type: (sqlite3.Connection, unicode) -> unicode
    """
    Create the collection inventory for the collection having this
    :class:`~pdart.pds4.LID` using the database connection and return
    it.
    """
    with closing(conn.cursor()) as cursor:
        lines = [u'P,%s\r\n' % str(product)
                 for (product,) in cast(Iterable[Tuple[unicode]],
                                        cursor.execute(
                    'SELECT product FROM products WHERE collection=?',
                    (collection_lid,)))]
    return ''.join(lines)


def make_db_collection_label_and_inventory(conn, lid, verify):
    # type: (sqlite3.Connection, unicode, bool) -> None
    """
    Create the label and inventory for the collection having this
    :class:`~pdart.pds4.LID` using the database connection, writing
    them to disk.  If verify is True, verify the label against its XML
    and Schematron schemas.  Raise an exception if either fails.
    """
    d = lookup_collection(conn, lid)
    label_fp = d['label_filepath']
    bundle = d['bundle']
    suffix = d['suffix']
    inventory_name = d['inventory_name']
    inventory_filepath = d['inventory_filepath']
    proposal_id = lookup_bundle(conn, bundle)['proposal_id']

    label = make_label({
            'lid': lid,
            'suffix': suffix,
            'proposal_id': str(proposal_id),
            'Citation_Information': placeholder_citation_information,
            'inventory_name': inventory_name
    }).toxml()
    label = pretty_print(label)

    with open(label_fp, 'w') as f:
        f.write(label)

    if verify:
        verify_label_or_raise(label)

    with io.open(inventory_filepath, 'wb', newline='') as f2:
        f2.write(make_db_collection_inventory(conn, lid))

    print 'collection label and inventory for', lid
    sys.stdout.flush()

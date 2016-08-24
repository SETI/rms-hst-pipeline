"""Functionality to build a bundle label using a SQLite database."""
from contextlib import closing
import sys

from pdart.pds4labels.BundleLabelXml import *
from pdart.pds4labels.DatabaseCaches import *
from pdart.xml.Schema import *


def make_db_bundle_label(conn, lid, verify):
    """
    Create the label text for the bundle having this
    :class:`~pdart.pds4.LID` using the database connection.  If verify
    is True, verify the label against its XML and Schematron schemas.
    Raise an exception if either fails.
    """
    d = lookup_bundle(conn, lid)
    label_fp = d['label_filepath']
    proposal_id = d['proposal_id']

    with closing(conn.cursor()) as cursor:
        reduced_collections = \
            [make_bundle_entry_member({'lid': collection_lid})
             for (collection_lid,)
             in cursor.execute(
                'SELECT collection from collections WHERE bundle=?', (lid,))]

    label = make_label({
            'lid': lid,
            'proposal_id': str(proposal_id),
            'Citation_Information': placeholder_citation_information,
            'Bundle_Member_Entries': combine_nodes_into_fragment(
                reduced_collections)
            }).toxml()
    with open(label_fp, 'w') as f:
        f.write(label)

    print 'bundle label for', lid
    sys.stdout.flush()

    if verify:
        verify_label_or_throw(label)

    return label

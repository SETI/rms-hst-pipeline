"""
Functionality to build a collection label using a SQLite database.
"""

from typing import TYPE_CHECKING

from pdart.new_labels.CitationInformation \
    import make_placeholder_citation_information
from pdart.new_labels.CollectionLabelXml import make_label
from pdart.new_labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


# TODO Should probably test document_collection independently.


def make_collection_label(bundle_db, collection_lidvid, verify):
    # type: (BundleDB, str, bool) -> unicode
    """
    Create the label text for the collection having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    collection_lid = lidvid_to_lid(collection_lidvid)
    collection_vid = lidvid_to_vid(collection_lidvid)
    collection = bundle_db.get_collection(collection_lidvid)
    suffix = collection.suffix
    proposal_id = bundle_db.get_bundle().proposal_id
    try:
        inventory_name = 'collection_%s.csv' % collection.prefix
    except:
        # Document collections won't have prefixes.
        inventory_name = 'collection.csv'

    label = make_label({
        'collection_lid': collection_lid,
        'collection_vid': collection_vid,
        'suffix': suffix,
        'proposal_id': str(proposal_id),
        'Citation_Information': make_placeholder_citation_information(
            collection_lid),
        'inventory_name': inventory_name
    }).toxml()

    return pretty_and_verify(label, verify)

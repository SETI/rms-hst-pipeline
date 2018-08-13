"""
Functionality to build a collection label using a SQLite database.
"""

from typing import TYPE_CHECKING

from pdart.new_labels.CitationInformation \
    import make_placeholder_citation_information
from pdart.new_labels.CollectionInventory import get_collection_inventory_name
from pdart.new_labels.CollectionLabelXml import make_label, \
    make_document_collection_title, make_non_document_collection_title
from pdart.new_labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


# TODO Should probably test document_collection independently.

def get_collection_label_name(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    collection = bundle_db.get_collection(collection_lidvid)
    try:
        collection_label_name = 'collection_%s.xml' % collection.prefix
    except Exception:
        # Document collections won't have prefixes.
        collection_label_name = 'collection.xml'
    return collection_label_name


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
    is_doc_coll = bundle_db.document_collection_exists(collection_lidvid)

    proposal_id = bundle_db.get_bundle().proposal_id
    if is_doc_coll:
        title = make_document_collection_title({
                'proposal_id': str(proposal_id)
                })
    else:
        title = make_non_document_collection_title({
                'suffix': collection.suffix,
                'proposal_id': str(proposal_id)})

    inventory_name = get_collection_inventory_name(bundle_db,
                                                   collection_lidvid)

    label = make_label({
        'collection_lid': collection_lid,
        'collection_vid': collection_vid,
        'title': title,
        'proposal_id': str(proposal_id),
        'Citation_Information': make_placeholder_citation_information(
            collection_lid),
        'inventory_name': inventory_name
    }).toxml()

    return pretty_and_verify(label, verify)

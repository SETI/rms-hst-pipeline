"""
Functionality to build a collection label using a SQLite database.
"""

from typing import TYPE_CHECKING

from pdart.new_labels.CitationInformation \
    import make_placeholder_citation_information
from pdart.new_labels.CollectionLabelXml import make_label
from pdart.new_labels.Utils import lidvid_to_lid
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


# TODO Should probably test document_collection independently.


def make_collection_label(bundle_db, collection_lidvid, verify):
    # type: (BundleDB, str, bool) -> unicode
    lid = lidvid_to_lid(collection_lidvid)
    collection = bundle_db.get_collection(collection_lidvid)
    suffix = collection.suffix
    proposal_id = bundle_db.get_bundle().proposal_id
    try:
        assert collection.prefix
        inventory_name = 'collection_%s.csv' % collection.prefix
    except:
        inventory_name = 'collection.csv'

    label = make_label({
        'lid': lid,
        'suffix': suffix,
        'proposal_id': str(proposal_id),
        'Citation_Information': make_placeholder_citation_information(lid),
        'inventory_name': inventory_name
    }).toxml()

    return pretty_and_verify(label, verify)


def make_collection_inventory(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> unicode
    products = bundle_db.get_collection_products(
        collection_lidvid)
    inventory_lines = [u'P,%s\r\n' % lidvid_to_lid(product.lidvid)
                       for product in products]
    return ''.join(inventory_lines)

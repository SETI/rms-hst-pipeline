"""
Functionality to build a collection label using a SQLite database.
"""
from typing import cast

from pdart.labels.CitationInformation import make_citation_information
from pdart.labels.CollectionInventory import get_collection_inventory_name
from pdart.labels.CollectionLabelXml import (
    make_label,
    make_document_collection_title,
    make_non_document_collection_title,
)
from pdart.labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

from pdart.citations import Citation_Information
from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import NonDocumentCollection, Collection, DocumentCollection


# TODO Should probably test document_collection independently.


def get_collection_label_name(bundle_db, collection_lidvid):
    # type: (BundleDB, str) -> str
    collection = bundle_db.get_collection(collection_lidvid)
    if isinstance(collection, DocumentCollection):
        # Document collections won't have prefixes.
        return "collection.xml"
    else:
        return "collection_%s.xml" % cast(NonDocumentCollection, collection).prefix


def make_collection_label(bundle_db, info, collection_lidvid, verify):
    # type: (BundleDB, Citation_Information, str, bool) -> bytes
    """
    Create the label text for the collection having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    collection_lid = lidvid_to_lid(collection_lidvid)
    collection_vid = lidvid_to_vid(collection_lidvid)
    collection: Collection = bundle_db.get_collection(collection_lidvid)
    is_doc_coll = bundle_db.document_collection_exists(collection_lidvid)

    proposal_id = bundle_db.get_bundle().proposal_id
    if is_doc_coll:
        title = make_document_collection_title({"proposal_id": str(proposal_id)})
    else:
        non_document_collection = cast(NonDocumentCollection, collection)
        title = make_non_document_collection_title(
            {"suffix": non_document_collection.suffix, "proposal_id": str(proposal_id)}
        )

    inventory_name = get_collection_inventory_name(bundle_db, collection_lidvid)

    label = (
        make_label(
            {
                "collection_lid": collection_lid,
                "collection_vid": collection_vid,
                "title": title,
                "proposal_id": str(proposal_id),
                "Citation_Information": make_citation_information(info),
                "inventory_name": inventory_name,
            }
        )
        .toxml()
        .encode()
    )

    return pretty_and_verify(label, verify)

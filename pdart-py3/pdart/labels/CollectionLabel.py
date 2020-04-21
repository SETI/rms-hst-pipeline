"""
Functionality to build a collection label using a SQLite database.
"""

from typing import cast, Callable

from pdart.citations import Citation_Information
from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    Collection,
    DocumentCollection,
    OtherCollection,
    switch_on_collection_subtype,
)
from pdart.labels.CitationInformation import make_citation_information
from pdart.labels.CollectionInventory import get_collection_inventory_name
from pdart.labels.CollectionLabelXml import (
    make_context_collection_title,
    make_document_collection_title,
    make_label,
    make_other_collection_title,
)
from pdart.labels.LabelError import LabelError
from pdart.labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify
from pdart.xml.Templates import NodeBuilder


# TODO Should probably test document_collection independently.


def get_collection_label_name(bundle_db: BundleDB, collection_lidvid: str) -> str:
    # We have to jump through some hoops to apply
    # switch_on_collection_type().
    def get_context_collection_label_name(collection: Collection) -> str:
        return "collection_context.xml"

    def get_document_collection_label_name(collection: Collection) -> str:
        return "collection.xml"

    def get_other_collection_label_name(collection: Collection) -> str:
        prefix = cast(OtherCollection, collection).prefix
        return f"collection_{prefix}.xml"

    collection: Collection = bundle_db.get_collection(collection_lidvid)
    return switch_on_collection_subtype(
        collection,
        get_context_collection_label_name,
        get_document_collection_label_name,
        get_other_collection_label_name,
    )(collection)


def make_collection_label(
    bundle_db: BundleDB,
    info: Citation_Information,
    collection_lidvid: str,
    verify: bool,
) -> bytes:
    """
    Create the label text for the collection having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    # TODO this is sloppy; is there a better way?
    products = bundle_db.get_collection_products(collection_lidvid)
    record_count = len(products)
    assert record_count > 0, f"{collection_lidvid} has no products"

    collection_lid = lidvid_to_lid(collection_lidvid)
    collection_vid = lidvid_to_vid(collection_lidvid)
    collection: Collection = bundle_db.get_collection(collection_lidvid)

    proposal_id = bundle_db.get_bundle().proposal_id

    def make_ctxt_coll_title(_coll: Collection) -> NodeBuilder:
        return make_context_collection_title({"proposal_id": str(proposal_id)})

    def make_doc_coll_title(_coll: Collection) -> NodeBuilder:
        return make_document_collection_title({"proposal_id": str(proposal_id)})

    def make_other_coll_title(coll: Collection) -> NodeBuilder:
        other_collection = cast(OtherCollection, coll)
        return make_other_collection_title(
            {"suffix": other_collection.suffix, "proposal_id": str(proposal_id)}
        )

    title: NodeBuilder = switch_on_collection_subtype(
        collection, make_ctxt_coll_title, make_doc_coll_title, make_other_coll_title,
    )(collection)

    inventory_name = get_collection_inventory_name(bundle_db, collection_lidvid)

    try:
        label = (
            make_label(
                {
                    "collection_lid": collection_lid,
                    "collection_vid": collection_vid,
                    "record_count": record_count,
                    "title": title,
                    "proposal_id": str(proposal_id),
                    "Citation_Information": make_citation_information(info),
                    "inventory_name": inventory_name,
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(str(e), collection_lidvid)

    return pretty_and_verify(label, verify)

"""Functionality to build a bundle label using a SQLite database."""

from typing import Dict, List

from pdart.citations import Citation_Information
from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    Collection,
    DocumentCollection,
    OtherCollection,
    switch_on_collection_subtype,
)
from pdart.labels.BundleLabelXml import (
    make_bundle_entry_member,
    make_label,
    make_bundle_context_node,
)
from pdart.labels.CitationInformation import make_citation_information
from pdart.labels.LabelError import LabelError
from pdart.labels.Utils import (
    lidvid_to_lid,
    lidvid_to_vid,
    create_target_identification_nodes,
    get_current_date,
    MOD_DATE_FOR_TESTESING,
)
from pdart.xml.Pretty import pretty_and_verify
from pdart.xml.Templates import (
    combine_nodes_into_fragment,
    NodeBuilder,
)


def make_bundle_label(
    bundle_db: BundleDB,
    bundle_lidvid: str,
    info: Citation_Information,
    verify: bool,
    use_mod_date_for_testing: bool = False,
) -> bytes:
    """
    Create the label text for the bundle in the bundle database using
    the database connection.  If verify is True, verify the label
    against its XML and Schematron schemas.  Raise an exception if
    either fails.
    """
    bundle = bundle_db.get_bundle(bundle_lidvid)
    proposal_id = bundle.proposal_id

    def get_ref_type(collection: Collection) -> str:
        return switch_on_collection_subtype(
            collection,
            "bundle_has_context_collection",
            "bundle_has_document_collection",
            "bundle_has_schema_collection",
            "bundle_has_data_collection",
        )

    reduced_collections = [
        make_bundle_entry_member(
            {
                "collection_lidvid": collection.lidvid,
                "ref_type": get_ref_type(collection),
            }
        )
        for collection in bundle_db.get_bundle_collections(bundle.lidvid)
    ]

    target_identifications = bundle_db.get_all_target_identification()
    target_identification_nodes: List[NodeBuilder] = []
    target_identification_nodes = create_target_identification_nodes(
        bundle_db, target_identifications, "bundle"
    )
    context_node: List[NodeBuilder] = []
    context_node = [make_bundle_context_node(target_identification_nodes)]

    if not use_mod_date_for_testing:
        # Get the date when the label is created
        mod_date = get_current_date()
    else:
        mod_date = MOD_DATE_FOR_TESTESING

    try:
        label = (
            make_label(
                {
                    "bundle_lid": lidvid_to_lid(bundle.lidvid),
                    "bundle_vid": lidvid_to_vid(bundle.lidvid),
                    "proposal_id": str(proposal_id),
                    "Citation_Information": make_citation_information(
                        info, is_for_bundle=True
                    ),
                    "mod_date": mod_date,
                    "Bundle_Member_Entries": combine_nodes_into_fragment(
                        reduced_collections
                    ),
                    "Context_Area": combine_nodes_into_fragment(context_node),
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(bundle.lidvid) from e

    assert label[:6] == b"<?xml ", "Not XML"
    return pretty_and_verify(label, verify)

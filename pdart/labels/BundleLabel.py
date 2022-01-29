"""Functionality to build a bundle label using a SQLite database."""

from typing import Any, Dict, List, cast

from pdart.citations import Citation_Information
from pdart.db.bundle_db import BundleDB
from pdart.db.sql_alch_tables import (
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
from pdart.labels.FitsProductLabelXml import (
    mk_Investigation_Area_lidvid,
    mk_Investigation_Area_name,
)
from pdart.labels.TimeCoordinates import get_time_coordinates
from pdart.labels.PrimaryResultSummary import primary_result_summary
from pdart.labels.InvestigationArea import investigation_area
from pdart.labels.ObservingSystem import observing_system
from pdart.labels.DocReferenceList import make_document_reference_list
from pdart.labels.CitationInformation import make_citation_information
from pdart.labels.LabelError import LabelError
from pdart.labels.utils import (
    lidvid_to_lid,
    lidvid_to_vid,
    get_current_date,
    MOD_DATE_FOR_TESTESING,
)
from pdart.labels.TargetIdentification import create_target_identification_nodes
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
        ref_type = switch_on_collection_subtype(
            collection,
            "bundle_has_context_collection",
            "bundle_has_document_collection",
            "bundle_has_schema_collection",
            "bundle_has_other_collection",
        )

        if ref_type == "bundle_has_other_collection":
            collection_type = cast(OtherCollection, collection).prefix
            ref_type = f"bundle_has_{collection_type}_collection"

        return ref_type

    reduced_collections = [
        make_bundle_entry_member(
            {
                "collection_lidvid": collection.lidvid,
                "ref_type": get_ref_type(collection),
            }
        )
        for collection in bundle_db.get_bundle_collections(bundle.lidvid)
    ]

    # Get the bundle title from part of CitationInformation description
    title = (
        info.title
        + ", HST Cycle "
        + str(info.cycle)
        + " Program "
        + str(info.propno)
        + ", "
        + info.publication_year
        + "."
    )

    # Get the list of target identifications nodes for the collection
    target_identifications = bundle_db.get_all_target_identification()
    target_identification_nodes: List[NodeBuilder] = []
    target_identification_nodes = create_target_identification_nodes(
        bundle_db, target_identifications, "bundle"
    )

    # Get the investigation node for the collection
    investigation_area_name = mk_Investigation_Area_name(proposal_id)
    investigation_area_lidvid = mk_Investigation_Area_lidvid(proposal_id)
    investigation_area_node = investigation_area(
        investigation_area_name, investigation_area_lidvid, "bundle"
    )

    # Get min start_time and max stop_time
    start_time, stop_time = bundle_db.get_roll_up_time_from_db()
    # Make sure start/stop time exists in db.
    if start_time is None:
        raise ValueError("Start time is not stored in FitsProduct table.")
    if stop_time is None:
        raise ValueError("Stop time is not stored in FitsProduct table.")

    start_stop_times = {
        "start_date_time": start_time,
        "stop_date_time": stop_time,
    }
    time_coordinates_node = get_time_coordinates(start_stop_times)

    # Dictionary used for primary result summary
    primary_result_dict: Dict[str, Any] = {}
    # Put dummy value in processing level, wait for update.
    primary_result_dict["processing_level"] = "Raw"
    instruments_list = bundle_db.get_instruments_of_the_bundle()
    instruments = ", ".join(instruments_list).upper()
    p_title = (
        f"{instruments} observations obtained by the HST "
        + f"Observing Program {proposal_id}."
    )
    primary_result_dict["description"] = p_title

    # Get unique wavelength names for roll-up in bundle
    wavelength_range = bundle_db.get_wavelength_range_from_db()
    primary_result_dict["wavelength_range"] = wavelength_range
    primary_result_summary_node = primary_result_summary(primary_result_dict)

    # Get the observing system node for the bundle
    observing_system_nodes: List[NodeBuilder] = [
        observing_system(instrument) for instrument in instruments_list
    ]

    context_node: List[NodeBuilder] = []
    context_node = [
        make_bundle_context_node(
            time_coordinates_node,
            primary_result_summary_node,
            investigation_area_node,
            observing_system_nodes,
            target_identification_nodes,
        )
    ]

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
                    "title": title,
                    "Citation_Information": make_citation_information(
                        info, is_for_bundle=True
                    ),
                    "mod_date": mod_date,
                    "Bundle_Member_Entries": combine_nodes_into_fragment(
                        reduced_collections
                    ),
                    "Context_Area": combine_nodes_into_fragment(context_node),
                    "Reference_List": make_document_reference_list(
                        instruments_list, "bundle"
                    ),
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(bundle.lidvid) from e

    if label[:6] != b"<?xml ":
        raise ValueError("Bundle label is not XML.")
    return pretty_and_verify(label, verify)

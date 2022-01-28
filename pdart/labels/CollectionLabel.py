"""
Functionality to build a collection label using a SQLite database.
"""

from typing import Any, cast, Callable, Dict, List

from pdart.citations import Citation_Information
from pdart.db.bundle_db import bundle_db
from pdart.db.sql_alch_tables import (
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
    make_schema_collection_title,
    make_collection_context_node,
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

from pdart.pipeline.SuffixInfo import get_processing_level  # type: ignore

# TODO Should probably test document_collection independently.


def get_collection_label_name(bundle_db: bundle_db, collection_lidvid: str) -> str:
    # We have to jump through some hoops to apply
    # switch_on_collection_type().
    def get_context_collection_label_name(collection: Collection) -> str:
        return "collection_context.xml"

    def get_document_collection_label_name(collection: Collection) -> str:
        return "collection.xml"

    def get_schema_collection_label_name(collection: Collection) -> str:
        return "collection_schema.xml"

    def get_other_collection_label_name(collection: Collection) -> str:
        collection_obj = cast(OtherCollection, collection)
        prefix = collection_obj.prefix
        instrument = collection_obj.instrument
        suffix = collection_obj.suffix
        return f"collection_{prefix}_{instrument}_{suffix}.xml"

    collection: Collection = bundle_db.get_collection(collection_lidvid)
    return switch_on_collection_subtype(
        collection,
        get_context_collection_label_name,
        get_document_collection_label_name,
        get_schema_collection_label_name,
        get_other_collection_label_name,
    )(collection)


def make_collection_label(
    bundle_db: bundle_db,
    info: Citation_Information,
    collection_lidvid: str,
    bundle_lidvid: str,
    verify: bool,
    use_mod_date_for_testing: bool = False,
) -> bytes:
    """
    Create the label text for the collection having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    collection = bundle_db.get_collection(collection_lidvid)

    # If a label is created for testing purpose to compare with pre-made XML
    # we will use MOD_DATE_FOR_TESTESING as the modification date.
    if not use_mod_date_for_testing:
        # Get the date when the label is created
        mod_date = get_current_date()
    else:
        mod_date = MOD_DATE_FOR_TESTESING

    return switch_on_collection_subtype(
        collection,
        make_context_collection_label,
        make_other_collection_label,
        make_schema_collection_label,
        make_other_collection_label,
    )(bundle_db, info, collection_lidvid, bundle_lidvid, verify, mod_date)


def make_context_collection_label(
    bundle_db: bundle_db,
    info: Citation_Information,
    collection_lidvid: str,
    bundle_lidvid: str,
    verify: bool,
    mod_date: str,
) -> bytes:
    """
    Create the label text for the ccontext ollection having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    # TODO this is sloppy; is there a better way?
    products = bundle_db.get_context_products()
    record_count = len(products)
    if record_count <= 0:
        raise ValueError(f"{collection_lidvid} has no context products.")

    collection_lid = lidvid_to_lid(collection_lidvid)
    collection_vid = lidvid_to_vid(collection_lidvid)
    collection: Collection = bundle_db.get_collection(collection_lidvid)

    proposal_id = bundle_db.get_bundle(bundle_lidvid).proposal_id
    instruments = ",".join(bundle_db.get_instruments_of_the_bundle()).upper()
    title: NodeBuilder = make_context_collection_title(
        {
            "instrument": instruments,
            "proposal_id": str(proposal_id),
        }
    )

    inventory_name = get_collection_inventory_name(bundle_db, collection_lidvid)

    try:
        label = (
            make_label(
                {
                    "collection_lid": collection_lid,
                    "collection_vid": collection_vid,
                    "record_count": record_count,
                    "title": title,
                    "mod_date": mod_date,
                    "proposal_id": str(proposal_id),
                    "Citation_Information": make_citation_information(info),
                    "inventory_name": inventory_name,
                    "Context_Area": combine_nodes_into_fragment([]),
                    "Reference_List": combine_nodes_into_fragment([]),
                    "collection_type": "Context",
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(collection_lidvid) from e

    return pretty_and_verify(label, verify)


def make_schema_collection_label(
    bundle_db: bundle_db,
    info: Citation_Information,
    collection_lidvid: str,
    bundle_lidvid: str,
    verify: bool,
    mod_date: str,
) -> bytes:
    """
    Create the label text for the schema collection having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    # TODO this is sloppy; is there a better way?
    products = bundle_db.get_schema_products()
    record_count = len(products)
    if record_count <= 0:
        raise ValueError(f"{collection_lidvid} has no schema products.")

    collection_lid = lidvid_to_lid(collection_lidvid)
    collection_vid = lidvid_to_vid(collection_lidvid)
    collection: Collection = bundle_db.get_collection(collection_lidvid)

    proposal_id = bundle_db.get_bundle(bundle_lidvid).proposal_id
    instruments = ",".join(bundle_db.get_instruments_of_the_bundle()).upper()
    title: NodeBuilder = make_schema_collection_title(
        {
            "instrument": instruments,
            "proposal_id": str(proposal_id),
        }
    )

    inventory_name = get_collection_inventory_name(bundle_db, collection_lidvid)

    try:
        label = (
            make_label(
                {
                    "collection_lid": collection_lid,
                    "collection_vid": collection_vid,
                    "record_count": record_count,
                    "title": title,
                    "mod_date": mod_date,
                    "proposal_id": str(proposal_id),
                    "Citation_Information": make_citation_information(info),
                    "inventory_name": inventory_name,
                    "Context_Area": combine_nodes_into_fragment([]),
                    "Reference_List": combine_nodes_into_fragment([]),
                    "collection_type": "Schema",
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(collection_lidvid) from e

    return pretty_and_verify(label, verify)


def make_other_collection_label(
    bundle_db: bundle_db,
    info: Citation_Information,
    collection_lidvid: str,
    bundle_lidvid: str,
    verify: bool,
    mod_date: str,
) -> bytes:
    """
    Create the label text for the document, browse, and data collection having
    this LIDVID using the bundle database.  If verify is True, verify the label
    against its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    # TODO this is sloppy; is there a better way?
    products = bundle_db.get_collection_products(collection_lidvid)
    record_count = len(products)
    if record_count <= 0:
        raise ValueError(f"{collection_lidvid} has no products.")

    collection_lid = lidvid_to_lid(collection_lidvid)
    collection_vid = lidvid_to_vid(collection_lidvid)
    collection: Collection = bundle_db.get_collection(collection_lidvid)

    proposal_id = bundle_db.get_bundle(bundle_lidvid).proposal_id
    instruments = ",".join(bundle_db.get_instruments_of_the_bundle()).upper()

    def make_ctxt_coll_title(_coll: Collection) -> NodeBuilder:
        return make_context_collection_title(
            {
                "instrument": instruments,
                "proposal_id": str(proposal_id),
            }
        )

    def make_doc_coll_title(_coll: Collection) -> NodeBuilder:
        return make_document_collection_title(
            {
                "instrument": instruments,
                "proposal_id": str(proposal_id),
            }
        )

    def make_sch_coll_title(_coll: Collection) -> NodeBuilder:
        return make_schema_collection_title(
            {
                "instrument": instruments,
                "proposal_id": str(proposal_id),
            }
        )

    def make_other_coll_title(coll: Collection) -> NodeBuilder:
        other_collection = cast(OtherCollection, coll)
        if other_collection.prefix == "browse":
            collection_title = (
                f"{other_collection.prefix.capitalize()} "
                + f"collection of {other_collection.instrument.upper()} "
                + f"observations obtained from HST Observing Program {proposal_id}."
            )
        else:
            # Get the data/misc collection title from db.
            collection_title = str(other_collection.title)
        return make_other_collection_title({"collection_title": collection_title})

    title: NodeBuilder = switch_on_collection_subtype(
        collection,
        make_ctxt_coll_title,
        make_doc_coll_title,
        make_sch_coll_title,
        make_other_coll_title,
    )(collection)

    inventory_name = get_collection_inventory_name(bundle_db, collection_lidvid)

    # Properly assign collection type for Document, Browse, or Data collection.
    # Context node only exists in Data collection label.
    # Reference_List only exists in Data collection label.
    context_node: List[NodeBuilder] = []
    reference_list_node: List[NodeBuilder] = []
    collection_type: str = ""
    type_name = type(collection).__name__
    if type_name == "DocumentCollection":
        collection_type = "Document"
        # For document collection, we need to add all handbooks in the csv but
        # we won't create the label for it.
        inst_list = bundle_db.get_instruments_of_the_bundle()
        record_count += 2 * len(inst_list)
    elif type_name == "OtherCollection":
        collection_type = cast(OtherCollection, collection).prefix.capitalize()
        suffix = cast(OtherCollection, collection).suffix
        instrument = cast(OtherCollection, collection).instrument

        # Roll-up (Context node) only exists in data collection
        if collection_type == "Data":
            # Get min start_time and max stop_time
            start_time, stop_time = bundle_db.get_roll_up_time_from_db(suffix)
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
            # Check if it's raw or calibrated image, we will update this later
            processing_level = get_processing_level(
                suffix=suffix, instrument_id=instrument
            )
            primary_result_dict["processing_level"] = processing_level

            p_title = bundle_db.get_fits_product_collection_title(collection_lidvid)
            primary_result_dict["description"] = p_title
            # Get unique wavelength names for roll-up in data collection
            wavelength_range = bundle_db.get_wavelength_range_from_db(suffix)
            primary_result_dict["wavelength_range"] = wavelength_range
            primary_result_summary_node = primary_result_summary(primary_result_dict)

            # Get the list of target identifications nodes for the collection
            target_identifications = bundle_db.get_all_target_identification()
            target_identification_nodes: List[NodeBuilder] = []
            target_identification_nodes = create_target_identification_nodes(
                bundle_db, target_identifications, "collection"
            )

            # Get the investigation node for the collection
            investigation_area_name = mk_Investigation_Area_name(proposal_id)
            investigation_area_lidvid = mk_Investigation_Area_lidvid(proposal_id)
            investigation_area_node = investigation_area(
                investigation_area_name, investigation_area_lidvid, "collection"
            )

            # Get the observing system node for the collection
            observing_system_node = observing_system(instrument)

            context_node = [
                make_collection_context_node(
                    time_coordinates_node,
                    primary_result_summary_node,
                    investigation_area_node,
                    observing_system_node,
                    target_identification_nodes,
                )
            ]

            # document reference list only exists in data collection
            reference_list_node = [
                make_document_reference_list([instrument], "collection")
            ]

    try:
        label = (
            make_label(
                {
                    "collection_lid": collection_lid,
                    "collection_vid": collection_vid,
                    "record_count": record_count,
                    "title": title,
                    "mod_date": mod_date,
                    "proposal_id": str(proposal_id),
                    "Citation_Information": make_citation_information(info),
                    "inventory_name": inventory_name,
                    "Context_Area": combine_nodes_into_fragment(context_node),
                    "collection_type": collection_type,
                    "Reference_List": combine_nodes_into_fragment(reference_list_node),
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(collection_lidvid) from e

    return pretty_and_verify(label, verify)

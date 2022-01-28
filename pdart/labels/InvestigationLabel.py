"""
Functionality to build a investigation label using a SQLite database.
"""

from typing import Any, Dict, List, Tuple

from pdart.db.bundle_db import bundle_db
from pdart.citations import Citation_Information
from pdart.labels.Lookup import Lookup
from pdart.labels.InvestigationLabelXml import (
    make_label,
    make_internal_ref,
    make_description,
)
from pdart.labels.FitsProductLabelXml import (
    mk_Investigation_Area_lid,
    mk_Investigation_Area_lidvid,
    mk_Investigation_Area_name,
)
from pdart.labels.utils import (
    get_current_date,
    MOD_DATE_FOR_TESTESING,
    lidvid_to_lid,
    lidvid_to_vid,
    date_time_to_date,
)
from pdart.labels.LabelError import LabelError
from pdart.xml.Pretty import pretty_and_verify
from pdart.xml.Templates import (
    NodeBuilder,
    combine_nodes_into_fragment,
)


def make_investigation_label(
    bundle_db: bundle_db,
    bundle_lidvid: str,
    info: Citation_Information,
    verify: bool,
    use_mod_date_for_testing: bool = False,
) -> bytes:
    """
    Create the label text for the context investigation having this LIDVID
    using the bundle database. If verify is True, verify the label against
    its XML and Schematron schemas. Raise an exception if either fails.
    """
    bundle = bundle_db.get_bundle(bundle_lidvid)
    proposal_id = bundle.proposal_id

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

    investigation_lid = mk_Investigation_Area_lid(proposal_id)
    investigation_lidvid = mk_Investigation_Area_lidvid(proposal_id)

    # Get min start_time and max stop_time
    start_time, stop_time = bundle_db.get_roll_up_time_from_db()
    # Make sure start/stop time exists in db.
    if start_time is None:
        raise ValueError("Start time is not stored in FitsProduct table.")
    if stop_time is None:
        raise ValueError("Stop time is not stored in FitsProduct table.")
    start_date = date_time_to_date(start_time)
    stop_date = date_time_to_date(stop_time)

    # internal_reference_nodes: List[NodeBuilder] = [make_alias(alias) for alias in alias_list]
    context_products = bundle_db.get_reference_context_products(investigation_lidvid)
    internal_reference_nodes: List[NodeBuilder] = []
    for product in context_products:
        ref_lid = lidvid_to_lid(product.lidvid)
        ref_type = f"investigation_to_{product.ref_type}"
        ref_node = make_internal_ref(ref_lid, ref_type)
        internal_reference_nodes.append(ref_node)

    description = info.abstract_formatted(indent=8)  # type: ignore
    if len(description) != 0:
        description = "\n".join(description)
    else:
        description = " " * 8 + "None"
    description_nodes: List[NodeBuilder] = [make_description(description)]

    if not use_mod_date_for_testing:
        # Get the date when the label is created
        mod_date = get_current_date()
    else:
        mod_date = MOD_DATE_FOR_TESTESING

    try:
        label = (
            make_label(
                {
                    "investigation_lid": investigation_lid,
                    "bundle_vid": lidvid_to_vid(bundle.lidvid),
                    "title": title,
                    "mod_date": mod_date,
                    "start_date": start_date,
                    "stop_date": stop_date,
                    "internal_reference": combine_nodes_into_fragment(
                        internal_reference_nodes
                    ),
                    "description": combine_nodes_into_fragment(description_nodes),
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(investigation_lid) from e

    return pretty_and_verify(label, verify)

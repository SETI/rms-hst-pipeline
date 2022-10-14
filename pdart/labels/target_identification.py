"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""

from typing import Any, Dict, List, Tuple

from pdart.db.bundle_db import BundleDB
from pdart.db.sql_alch_tables import TargetIdentification
from pdart.labels.lookup import Lookup
from pdart.labels.target_identification_xml import (
    approximate_target_table,
    target_identification,
    get_target_lid,
    get_target_lidvid,
    get_target_lidvid_by_lid,
    make_label,
    make_alias,
    make_alias_list,
    make_description,
)
from pdart.labels.utils import (
    get_current_date,
    MOD_DATE_FOR_TESTESING,
)
from pdart.labels.label_error import LabelError
from pdart.xml.pretty import pretty_and_verify
from pdart.xml.templates import (
    NodeBuilder,
    combine_nodes_into_fragment,
)

_USING_PLACEHOLDER: bool = True

if _USING_PLACEHOLDER:
    _PLACEHOLDER_NAME = "Magrathea"
    _PLACEHOLDER_TYPE = "Planet"
    _PLACEHOLDER: Dict[str, str] = {
        "name": _PLACEHOLDER_NAME,
        "type": _PLACEHOLDER_TYPE,
        "description": f"The {_PLACEHOLDER_TYPE.lower()} {_PLACEHOLDER_NAME}",
        "lid": get_target_lid([_PLACEHOLDER_TYPE, _PLACEHOLDER_NAME]),
    }


def get_target_info(lookup: Lookup) -> Dict[str, str]:
    try:
        targname = lookup["TARGNAME"]
    except KeyError as e:
        raise ValueError(f"No value for TARGNAME in {lookup}") from e

    for prefix, info in approximate_target_table.items():
        if targname.startswith(prefix):
            if len(info) not in [2, 3]:
                raise ValueError(f"unexpected target_info: {info}.")
            name = info[0]
            type = info[1]
            if len(info) == 2:
                return {
                    "name": name,
                    "type": type,
                    "description": f"The {type.lower()} {name}",
                    "lid": get_target_lid([type, name]),
                    "alternate_designations": "",
                }
            elif len(info) == 3:
                primary = info[2]
                return {
                    "name": name,
                    "type": type,
                    "description": f"The {type.lower()} of {primary}, {name}",
                    "lid": get_target_lid([type, primary, name]),
                    "alternate_designations": "",
                }

    if _USING_PLACEHOLDER:
        return _PLACEHOLDER
    else:
        raise ValueError(f"TARGNAME {targname} doesn't match approximations")


def get_target(target_info: Dict[str, str]) -> NodeBuilder:
    """
    Given the FITS header fields for a product, create a
    ``<Target_Identification />`` XML element using heuristics.
    """
    return target_identification(
        target_info["name"],
        target_info["type"],
        target_info["alternate_designations"],
        target_info["description"],
        target_info["lid"],
        target_info["reference_type"],
    )


def create_target_identification_nodes(
    bundle_db: BundleDB,
    target_identifications: List[TargetIdentification],
    reference_type: str,
) -> List[NodeBuilder]:
    """
    Take in a list of TargetIdentification records (from db), build a target
    dictionary for each record and return a list of Target_Identification nodes.
    It will be inserted in XML when passing into make_label.
    """
    reference_type_dict = {
        "data": "data_to_target",
        "collection": "collection_to_target",
        "bundle": "bundle_to_target",
    }
    target_identification_nodes: List[NodeBuilder] = []
    for target in target_identifications:
        target_lidvid = get_target_lidvid_by_lid(target.lid_reference)
        bundle_db.create_context_product(target_lidvid, "target")
        # bundle_db.create_context_product(
        #     get_target_lidvid([target.type, target.name]), "target"
        # )
        target_dict: Dict[str, Any] = {}
        target_dict["name"] = target.name
        target_dict["type"] = target.type
        target_dict["alternate_designations"] = target.alternate_designations
        target_dict["description"] = target.description
        target_dict["lid"] = target.lid_reference
        target_dict["reference_type"] = reference_type_dict[reference_type]
        target_identification_nodes.append(get_target(target_dict))
    return target_identification_nodes


def make_context_target_label(
    bundle_db: BundleDB,
    target_info: Dict[str, str],
    verify: bool,
    use_mod_date_for_testing: bool = False,
) -> bytes:
    """
    Create the label text for the context target having this LIDVID using
    the bundle database.  If verify is True, verify the label against
    its XML and Schematron schemas.  Raise an exception if either
    fails.
    """
    target_id = target_info["target_id"]
    target_lid = target_info["target_lid"]
    target_lidvid = f"{target_lid}::1.0"
    # For the case when we can't search by target_lid, ex: 8218, the lid reference
    # reference stored in db is "urn:nasa:pds:context:target:satellite.neptune.triton"
    # which can't be searched by target_id "urn:nasa:pds:context:target:satellite.triton"
    # we will search by target_id.
    try:
        target_identification = bundle_db.get_target_identification_based_on_lid(
            target_lid
        )
    except:
        target_identification = bundle_db.get_target_identification_based_on_id_and_lid(
            target_id, target_lid
        )
    target_lid = target_identification.lid_reference
    bundle_db.create_context_product(
        get_target_lidvid_by_lid(target_lid),
        "target",
    )

    alias = str(target_identification.alternate_designations)
    alias_list_nodes: List[NodeBuilder] = []
    if len(alias) != 0:
        alias_list = alias.split("\n")
        alias_list_nodes = [make_alias_list(alias_list)]

    target_description = str(target_identification.description)
    if len(target_description) != 0:
        # properly align multi line textnodes with 8 spaces
        target_description = " " * 8 + target_description
        target_description = target_description.replace("\n", "\n" + " " * 8)
    else:
        target_description = " " * 8 + "None"
    description_nodes: List[NodeBuilder] = [make_description(target_description)]

    if not use_mod_date_for_testing:
        # Get the date when the label is created
        mod_date = get_current_date()
    else:
        mod_date = MOD_DATE_FOR_TESTESING

    try:
        label = (
            make_label(
                {
                    "target_lid": target_lid,
                    "target_vid": "1.0",
                    "title": target_identification.name,
                    "alias_list": combine_nodes_into_fragment(alias_list_nodes),
                    "name": target_identification.name,
                    "type": target_identification.type,
                    "description": combine_nodes_into_fragment(description_nodes),
                    "mod_date": mod_date,
                }
            )
            .toxml()
            .encode()
        )
    except Exception as e:
        raise LabelError(target_lidvid) from e

    return pretty_and_verify(label, verify)

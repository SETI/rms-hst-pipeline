"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""

from typing import Any, Dict, List, Tuple

from pdart.db.BundleDB import BundleDB
from pdart.labels.TargetIdentificationXml import (
    approximate_target_table,
    target_identification,
    target_lid,
)
from pdart.xml.Templates import NodeBuilder

_USING_PLACEHOLDER: bool = True


def get_target_info(card_dictionaries: List[Dict[str, Any]]) -> Dict[str, str]:
    targname = card_dictionaries[0]["TARGNAME"]

    for prefix, (name, type) in approximate_target_table.items():
        if targname.startswith(prefix):
            return {
                "name": name,
                "type": type,
                "description": f"The {type.lower()} {name}",
                "lid": target_lid(name, type),
            }
    if _USING_PLACEHOLDER:
        # TODO-PLACEHOLER
        name = "Magrathea"
        type = "Planet"
        return {
            "name": name,
            "type": type,
            "description": f"The {type.lower()} {name}",
            "lid": target_lid(name, type),
        }

    raise ValueError(f"TARGNAME {targname} doesn't match approximations")


def get_target(target_info: Dict[str, str]) -> NodeBuilder:
    """
    Given the FITS header fields for a product, create a
    ``<Target_Identification />`` XML element using heuristics.
    """
    return target_identification(
        target_info["name"],
        target_info["type"],
        target_info["description"],
        target_info["lid"],
    )

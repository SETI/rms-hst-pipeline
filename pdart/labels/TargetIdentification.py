"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""

from typing import Any, Dict, List, Tuple

from pdart.db.BundleDB import BundleDB
from pdart.labels.Lookup import Lookup
from pdart.labels.TargetIdentificationXml import (
    approximate_target_table,
    target_identification,
    target_lid,
)
from pdart.xml.Templates import NodeBuilder

_USING_PLACEHOLDER: bool = True


def get_target_info(lookup: Lookup) -> Dict[str, str]:
    targname = lookup["TARGNAME"]

    for prefix, info in approximate_target_table.items():
        if targname.startswith(prefix):
            assert len(info) in [2, 3], f"unexpected target_info: {info}"
            name = info[0]
            type = info[1]
            if len(info) == 2:
                return {
                    "name": name,
                    "type": type,
                    "description": f"The {type.lower()} {name}",
                    "lid": target_lid([type, name]),
                }
            elif len(info) == 3:
                primary = info[2]
                return {
                    "name": name,
                    "type": type,
                    "description": f"The {type.lower()} of {primary}, {name}",
                    "lid": target_lid([type, primary, name]),
                }

    if _USING_PLACEHOLDER:
        # TODO-PLACEHOLER
        name = "Magrathea"
        type = "Planet"
        return {
            "name": name,
            "type": type,
            "description": f"The {type.lower()} {name}",
            "lid": target_lid([type, name]),
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

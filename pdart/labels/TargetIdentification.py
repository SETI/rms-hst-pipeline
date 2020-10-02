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

if _USING_PLACEHOLDER:
    _PLACEHOLDER_NAME = "Magrathea"
    _PLACEHOLDER_TYPE = "Planet"
    _PLACEHOLDER: Dict[str, str] = {
        "name": _PLACEHOLDER_NAME,
        "type": _PLACEHOLDER_TYPE,
        "description": f"The {_PLACEHOLDER_TYPE.lower()} {_PLACEHOLDER_NAME}",
        "lid": target_lid([_PLACEHOLDER_TYPE, _PLACEHOLDER_NAME]),
    }


def get_target_info(lookup: Lookup) -> Dict[str, str]:
    try:
        targname = lookup["TARGNAME"]
    except KeyError as e:
        if _USING_PLACEHOLDER:
            return _PLACEHOLDER
        else:
            raise ValueError(f"No value for TARGNAME in {lookup}") from e

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
        target_info["description"],
        target_info["lid"],
    )

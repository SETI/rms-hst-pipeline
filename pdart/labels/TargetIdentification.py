"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""

from pdart.labels.TargetIdentificationXml import (
    approximate_target_table,
    target_identification,
)
from typing import Any, Callable, Dict, List, Tuple
from pdart.xml.Templates import NodeBuilder


def _get_target(card_dictionaries: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    targname = card_dictionaries[0]["TARGNAME"]

    for prefix, (name, type) in approximate_target_table.items():
        if targname.startswith(prefix):
            return (name, type, f"The {type.lower()} {name}")
    raise Exception(f"TARGNAME {targname} doesn't match approximations")


def get_target(card_dictionaries: List[Dict[str, Any]]) -> NodeBuilder:
    """
    Given the FITS header fields for a product, create a
    ``<Target_Identification />`` XML element using heuristics.
    """
    return target_identification(*(_get_target(card_dictionaries)))

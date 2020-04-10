"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""
from typing import TYPE_CHECKING

from pdart.new_labels.TargetIdentificationXml import \
    approximate_target_table, get_placeholder_target, target_identification
from pdart.rules.Combinators import multiple_implementations

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Tuple
    from pdart.xml.Templates import NodeBuilder


def _get_target_from_header_unit(card_dictionaries):
    # type: (List[Dict[str, Any]]) -> Tuple[unicode, unicode, unicode]
    targname = card_dictionaries[0]['TARGNAME']

    for prefix, (name, type) in approximate_target_table.iteritems():
        if targname.startswith(prefix):
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


_get_target = multiple_implementations('_get_target',
                                       _get_target_from_header_unit,
                                       get_placeholder_target
)  # type: Callable[[List[Dict[str, Any]]], Tuple[unicode, unicode, unicode]]


def get_target(card_dictionaries):
    # type: (List[Dict[str, Any]]) -> NodeBuilder
    """
    Given the FITS header fields for a product, create a
    ``<Target_Identification />`` XML element using heuristics.
    """
    return target_identification(*(_get_target(card_dictionaries)))

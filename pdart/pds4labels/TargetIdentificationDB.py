"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""
from pdart.pds4labels.TargetIdentificationXml import *
from pdart.rules.Combinators import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pdart.pds4labels.DBCalls import Headers


def _db_get_target_from_header_unit(headers):
    # type: (Headers) -> Tuple[unicode, unicode, unicode]
    targname = headers[0]['TARGNAME']

    for prefix, (name, type) in approximate_target_table.iteritems():
        if targname.startswith(prefix):
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


_get_db_target = multiple_implementations('_get_db_target',
                                          _db_get_target_from_header_unit,
                                          get_placeholder_target)
# type: Callable[[Headers], Tuple[unicode, unicode, unicode]]


def get_db_target(headers):
    # type: (Headers) -> NodeBuilder
    """
    Given the FITS header fields for a product, create a
    ``<Target_Identification />`` XML element using heuristics.
    """
    return target_identification(*(_get_db_target(headers)))

"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a SQLite database.
"""
from pdart.pds4labels.TargetIdentificationXml import *
from pdart.rules.Combinators import *


def _db_get_target_from_header_unit(headers):
    # type: (List[Dict[str, Any]]) -> Tuple[unicode, unicode, unicode]
    targname = headers[0]['TARGNAME']

    for prefix, (name, type) in approximate_target_table.iteritems():
        if targname.startswith(prefix):
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


_get_db_target = multiple_implementations('_get_db_target',
                                          _db_get_target_from_header_unit,
                                          get_placeholder_target)


def get_db_target(headers):
    # type: (List[Dict[str, Any]]) -> NodeBuilder
    """
    Given the FITS header fields for a product, create a
    ``<Target_Identification />`` XML element using heuristics.
    """
    return target_identification(*(_get_db_target(headers)))

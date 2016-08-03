from pdart.exceptions.Combinators import *
from pdart.pds4labels.TargetIdentificationXml import *


def _db_get_target_from_header_unit(headers):
    targname = headers[0]['TARGNAME']

    for prefix, (name, type) in _approximate_target_table.iteritems():
        if targname.startswith(prefix):
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


_get_db_target = multiple_implementations('_get_db_target',
                                          _db_get_target_from_header_unit,
                                          get_placeholder_target)


def get_db_target(headers, conn, lid):
    return target_identification(*(_get_db_target(headers)))

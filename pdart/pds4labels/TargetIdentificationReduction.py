"""
Functionality to build a ``<Target_Identification />`` XML element of
a product label using a
:class:`~pdart.reductions.Reduction.Reduction`.
"""
from pdart.exceptions.Combinators import *
from pdart.reductions.Reduction import *
from pdart.pds4labels.TargetIdentificationXml import *


def _get_target_from_header_unit(header_unit):
    targname = header_unit['TARGNAME']
    for prefix, (name, type) in _approximate_target_table.iteritems():
        if targname.startswith(prefix):
            return (name, type, 'The %s %s' % (type.lower(), name))
    raise Exception('TARGNAME %s doesn\'t match approximations' % targname)


_get_target = multiple_implementations('_get_target',
                                       _get_target_from_header_unit,
                                       get_placeholder_target)


class TargetIdentificationReduction(Reduction):
    """Reduce a product to an XML Target_Identification node template."""
    def reduce_fits_file(self, file, get_reduced_hdus):
        # Doc -> Node
        get_target = multiple_implementations('get_target',
                                              lambda: get_reduced_hdus()[0],
                                              get_placeholder_target)
        return target_identification(*get_target())

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # tuple or None
        if n == 0:
            return get_reduced_header_unit()
        else:
            pass

    def reduce_header_unit(self, n, header_unit):
        # tuple or None
        if n == 0:
            return _get_target(header_unit)
        else:
            pass

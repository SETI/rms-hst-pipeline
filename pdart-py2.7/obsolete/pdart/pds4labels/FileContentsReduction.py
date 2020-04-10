"""
Functionality to build the XML fragment containing the needed
``<Header />`` and ``<Array />`` or ``<Array_2D_Image />`` elements of
a product label using a :class:`~pdart.reductions.Reduction.Reduction`.
"""
from pdart.pds4labels.FileContentsXml import *
from pdart.reductions.Reduction import *
from pdart.xml.Templates import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List
    # TODO Make mypy stubs for pyfits
    _HDU = Any


def _mk_axis_arrays(hdu, axes):
    # type: (List[_HDU], int) -> FragBuilder
    def mk_axis_array(hdu, i):
        # type: (_HDU, int) -> NodeBuilder

        axis_name = AXIS_NAME_TABLE[i]
        elements = str(hdu.header['NAXIS%d' % i])
        # TODO Check the semantics of sequence_number
        sequence_number = str(i)
        return axis_array({'axis_name': axis_name,
                           'elements': elements,
                           'sequence_number': sequence_number})

    return combine_nodes_into_fragment(
        [mk_axis_array(hdu, i + 1) for i in range(0, axes)])


class FileContentsReduction(Reduction):
    """
    Reduce a product to an XML fragment template containing the
    ``<Header />`` and ``<Array_2D_Image />`` elements describing its
    contents.
    """

    def reduce_fits_file(self, file, get_reduced_hdus):
        # Doc -> Fragment
        reduced_hdus = get_reduced_hdus()
        return combine_fragments_into_fragment(reduced_hdus)

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        # Doc -> Fragment
        local_identifier = 'hdu_%d' % n
        fileinfo = hdu.fileinfo()
        offset = str(fileinfo['hdrLoc'])
        object_length = str(fileinfo['datLoc'] - fileinfo['hdrLoc'])
        header = header_contents({'local_identifier': local_identifier,
                                  'offset': offset,
                                  'object_length': object_length})

        if fileinfo['datSpan']:
            axes = hdu.header['NAXIS']
            data_type = BITPIX_TABLE[hdu.header['BITPIX']]
            elmt_arr = element_array({'data_type': data_type})

            if axes == 1:
                data = data_1d_contents({
                        'offset': str(fileinfo['datLoc']),
                        'Element_Array': elmt_arr,
                        'Axis_Arrays': _mk_axis_arrays(hdu, axes)
                        })
            elif axes == 2:
                data = data_2d_contents({
                        'offset': str(fileinfo['datLoc']),
                        'Element_Array': elmt_arr,
                        'Axis_Arrays': _mk_axis_arrays(hdu, axes)
                        })
            node_functions = [header, data]
        else:
            node_functions = [header]

        return combine_nodes_into_fragment(node_functions)

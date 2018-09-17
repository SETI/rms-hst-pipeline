"""
Functionality to build the XML fragment containing the needed
``<Header />`` and ``<Array />`` or ``<Array_2D_Image />`` elements of
a product label using a SQLite database.
"""
from typing import TYPE_CHECKING

from pdart.new_db.BundleDB import BundleDB
from pdart.new_db.FitsFileDB import get_file_offsets
from pdart.new_labels.FileContentsXml import AXIS_NAME_TABLE, BITPIX_TABLE, \
    axis_array, data_1d_contents, data_2d_contents, data_3d_contents, \
    element_array, header_contents
from pdart.xml.Templates import combine_fragments_into_fragment, \
    combine_nodes_into_fragment

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List
    from pdart.xml.Templates import FragBuilder, NodeBuilder


def _mk_axis_arrays(card_dicts, hdu_index, axes):
    # type: (List[Dict[str, Any]], int, int) -> FragBuilder

    def mk_axis_array(i):
        # type: (int) -> NodeBuilder
        axis_name = AXIS_NAME_TABLE[i]

        elements = card_dicts[hdu_index]['NAXIS%d' % i]
        # TODO Check the semantics of sequence_number
        sequence_number = str(i)
        return axis_array({'axis_name': axis_name,
                           'elements': str(elements),
                           'sequence_number': sequence_number})

    return combine_nodes_into_fragment(
        [mk_axis_array(i + 1) for i in range(0, axes)])


def get_file_contents(bundle_db, card_dicts, fits_product_lidvid):
    # type: (BundleDB, List[Dict[str, Any]], unicode) -> FragBuilder
    """
    Given the dictionary of the header fields from a product's FITS
    file, an open connection to the database, and the product's
    :class:`~pdart.pds4.LIDVID`, return an XML fragment containing the
    needed ``<Header />`` and ``<Array />`` or ``<Array_2D_Image />``
    elements for the FITS file's HDUs.
    """

    def get_hdu_contents(hdu_index, hdrLoc, datLoc, datSpan):
        # type: (int, int, int, int) -> FragBuilder
        """
        Return an XML fragment containing the needed ``<Header />``
        and ``<Array />`` or ``<Array_2D_Image />`` elements for the
        FITS file's HDUs.
        """
        local_identifier = 'hdu_%d' % hdu_index
        offset = str(hdrLoc)
        object_length = str(datLoc - hdrLoc)
        header = header_contents({'local_identifier': local_identifier,
                                  'offset': offset,
                                  'object_length': object_length})

        if datSpan:
            bitpix = int(card_dicts[hdu_index]['BITPIX'])
            axes = int(card_dicts[hdu_index]['NAXIS'])
            data_type = BITPIX_TABLE[bitpix]
            elmt_arr = element_array({'data_type': data_type})

            assert axes in [1, 2, 3], \
                ('NAXIS = %d in hdu #%d in %s' %
                 (axes, hdu_index, fits_product_lidvid))
            if axes == 1:
                data = data_1d_contents({
                    'offset': str(datLoc),
                    'Element_Array': elmt_arr,
                    'Axis_Arrays': _mk_axis_arrays(card_dicts,
                                                   hdu_index, axes)
                })
            elif axes == 2:
                data = data_2d_contents({
                    'offset': str(datLoc),
                    'Element_Array': elmt_arr,
                    'Axis_Arrays': _mk_axis_arrays(card_dicts,
                                                   hdu_index, axes)
                })
            elif axes == 3:
                data = data_3d_contents({
                    'offset': str(datLoc),
                    'Element_Array': elmt_arr,
                    'Axis_Arrays': _mk_axis_arrays(card_dicts,
                                                   hdu_index, axes)
                })
            node_functions = [header, data]
        else:
            node_functions = [header]

        return combine_nodes_into_fragment(node_functions)

    return combine_fragments_into_fragment(
        [get_hdu_contents(*hdu)
         for hdu in get_file_offsets(bundle_db, fits_product_lidvid)])

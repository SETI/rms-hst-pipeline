"""
Functionality to build the XML fragment containing the needed
``<Header />`` and ``<Array />`` or ``<Array_2D_Image />`` elements of
a product label using a SQLite database.
"""
from contextlib import closing

from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.FileContentsXml import *

from typing import Callable, cast, Dict, Iterable, List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3


def _db_mk_axis_arrays(headers, hdu_index, axes):
    # type: (List[Dict[str, Any]], int, int) -> FragBuilder
    def mk_axis_array(i):
        axis_name = AXIS_NAME_TABLE[i]

        elements = headers[hdu_index]['NAXIS%d' % i]
        # TODO Check the semantics of sequence_number
        sequence_number = str(i)
        return axis_array({'axis_name': axis_name,
                           'elements': str(elements),
                           'sequence_number': sequence_number})

    return combine_nodes_into_fragment(
        [mk_axis_array(i + 1) for i in range(0, axes)])


def get_db_file_contents(headers, conn, lid):
    # type: (List[Dict[str, Any]], sqlite3.Connection, unicode) -> FragBuilder
    """
    Given the dictionary of the header fields from a product's FITS
    file, an open connection to the database, and the product's
    :class:`~pdart.pds4.LID`, return an XML fragment containing the
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
            bitpix = headers[hdu_index]['BITPIX']
            axes = headers[hdu_index]['NAXIS']
            data_type = BITPIX_TABLE[bitpix]
            elmt_arr = element_array({'data_type': data_type})

            if axes == 1:
                data = data_1d_contents({
                        'offset': str(datLoc),
                        'Element_Array': elmt_arr,
                        'Axis_Arrays': _db_mk_axis_arrays(headers,
                                                          hdu_index, axes)
                        })
            elif axes == 2:
                data = data_2d_contents({
                        'offset': str(datLoc),
                        'Element_Array': elmt_arr,
                        'Axis_Arrays': _db_mk_axis_arrays(headers,
                                                          hdu_index, axes)
                        })
            node_functions = [header, data]
        else:
            node_functions = [header]

        return combine_nodes_into_fragment(node_functions)

    with closing(conn.cursor()) as cursor:
        return combine_fragments_into_fragment(
            [get_hdu_contents(*hdu)
             for hdu in get_fits_file_offsets_db(cursor, lid)])

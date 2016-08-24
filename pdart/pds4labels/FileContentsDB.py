"""
Functionality to build the XML fragment containing the needed
``<Header />`` and ``<Array />`` or ``<Array_2D_Image />`` elements of
a product label using a SQLite database.
"""
from contextlib import closing

from pdart.pds4labels.FileContentsXml import *


def _db_mk_axis_arrays(headers, hdu_index, axes):
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
    """
    Given the dictionary of the header fields from a product's FITS
    file, an open connection to the database, and the product's
    :class:`~pdart.pds4.LID`, return an XML fragment containing the
    needed ``<Header />`` and ``<Array />`` or ``<Array_2D_Image />``
    elements for the FITS file's HDUs.
    """
    def get_hdu_contents(hdu_index, hdrLoc, datLoc, datSpan):
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

        assert is_doc_to_node_function(header)

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
            assert is_doc_to_node_function(data)
            node_functions = [header, data]
        else:
            node_functions = [header]

        res = combine_nodes_into_fragment(node_functions)
        assert is_doc_to_fragment_function(res)
        return res

    with closing(conn.cursor()) as cursor:
        return combine_fragments_into_fragment(
            [get_hdu_contents(*hdu)
             for hdu in cursor.execute(
                    """SELECT hdu_index, hdrLoc, datLoc, datSpan
                       FROM hdus WHERE product=?
                       ORDER BY hdu_index ASC""",
                    (str(lid),))])

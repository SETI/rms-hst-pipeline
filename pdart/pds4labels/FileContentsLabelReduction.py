from pdart.reductions.Reduction import *
from pdart.xml.Templates import *


# For product labels: produces the fragment of the File node that
# contains Header and Array_2D_Image elements.


_AXIS_NAME_TABLE = {
    1: 'Line',
    2: 'Sample'
    }

_BITPIX_TABLE = {
    # TODO Verify these
    8: 'UnsignedByte',
    16: 'SignedMSB2',
    32: 'SignedMSB4',
    64: 'SignedMSB8',
    -32: 'IEEE754MSBSingle',
    -62: 'IEEE754MSBDouble'
    }


axis_array = interpret_template("""<Axis_Array>
<axis_name><NODE name="axis_name"/></axis_name>
<elements><NODE name="elements"/></elements>
<sequence_number><NODE name="sequence_number"/></sequence_number>
</Axis_Array>""")


def mk_axis_arrays(hdu, axes):
    def mk_axis_array(hdu, i):
        axis_name = _AXIS_NAME_TABLE[i]
        elements = str(hdu.header['NAXIS%d' % i])
        # TODO Check the semantics of sequence_number
        sequence_number = str(i)
        return axis_array({'axis_name': axis_name,
                           'elements': elements,
                           'sequence_number': sequence_number})

    return combine_nodes_into_fragment(
        [mk_axis_array(hdu, i) for i in range(1, axes + 1)])

header_contents = interpret_template("""<Header>
<local_identifier><NODE name="local_identifier"/></local_identifier>
<offset unit="byte"><NODE name="offset"/></offset>
<object_length unit="byte"><NODE name="object_length"/></object_length>
<parsing_standard_id>FITS 3.0</parsing_standard_id>
<description>Global FITS Header</description>
</Header>""")

data_contents = interpret_template("""<Array_2D_Image>
<offset unit="byte"><NODE name="offset" /></offset>
<axes><NODE name="axes" /></axes>
<axis_index_order>Last Index Fastest</axis_index_order>
<NODE name="Element_Array" />
<FRAGMENT name="Axis_Arrays" />
</Array_2D_Image>""")

element_array = interpret_template("""<ElementArray>
<data_type><NODE name="data_type" /></data_type></ElementArray>""")


class FileContentsLabelReduction(Reduction):
    def reduce_fits_file(self, file, get_reduced_hdus):
        reduced_hdus = get_reduced_hdus()
        return combine_fragments_into_fragment(reduced_hdus)

    def reduce_hdu(self, n, hdu,
                   get_reduced_header_unit,
                   get_reduced_data_unit):
        local_identifier = 'hdu_%d' % n
        fileinfo = hdu.fileinfo()
        offset = str(fileinfo['hdrLoc'])
        object_length = str(fileinfo['datLoc'] - fileinfo['hdrLoc'])
        header = header_contents({'local_identifier': local_identifier,
                                  'offset': offset,
                                  'object_length': object_length})
        assert is_doc_to_node_function(header)

        if fileinfo['datSpan']:
            axes = hdu.header['NAXIS']
            data_type = _BITPIX_TABLE[hdu.header['BITPIX']]
            elmt_arr = element_array({'data_type': data_type})

            data = data_contents({
                    'offset': str(fileinfo['datLoc']),
                    'axes': str(axes),
                    'Element_Array': elmt_arr,
                    'Axis_Arrays': mk_axis_arrays(hdu, axes)
                    })
            assert is_doc_to_node_function(data)
            node_functions = [header, data]
        else:
            node_functions = [header]

        res = combine_nodes_into_fragment(node_functions)
        assert is_doc_to_fragment_function(res)
        return res

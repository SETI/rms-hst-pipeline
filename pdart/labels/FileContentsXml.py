"""
Templates to create a XML fragment containing the needed ``<Header
/>`` and ``<Array />`` or ``<Array_2D_Image />`` elements of a product
label.
"""

from typing import Dict

from pdart.xml.Templates import NodeBuilderTemplate, interpret_template

# For product labels: produces the fragment of the File node that
# contains Header and Array_2D_Image elements.


AXIS_NAME_TABLE: Dict[int, str] = {1: "Line", 2: "Sample", 3: "Band"}


BITPIX_TABLE: Dict[int, str] = {
    # TODO Verify these
    8: "UnsignedByte",
    16: "SignedMSB2",
    32: "SignedMSB4",
    64: "SignedMSB8",
    -32: "IEEE754MSBSingle",
    -62: "IEEE754MSBDouble",
}


axis_array: NodeBuilderTemplate = interpret_template(
    """<Axis_Array>
<axis_name><NODE name="axis_name"/></axis_name>
<elements><NODE name="elements"/></elements>
<sequence_number><NODE name="sequence_number"/></sequence_number>
</Axis_Array>"""
)
"""
An interpreted node template to create an ``<Axis_Array />``
XML element.
"""

header_contents: NodeBuilderTemplate = interpret_template(
    """<Header>
<local_identifier><NODE name="local_identifier"/></local_identifier>
<offset unit="byte"><NODE name="offset"/></offset>
<object_length unit="byte"><NODE name="object_length"/></object_length>
<parsing_standard_id>FITS 3.0</parsing_standard_id>
<description><NODE name="description"/></description>
</Header>"""
)
"""
An interpreted node template to create a ``<Header />``
XML element.
"""

data_1d_contents: NodeBuilderTemplate = interpret_template(
    """<Array>
<offset unit="byte"><NODE name="offset" /></offset>
<axes>1</axes>
<axis_index_order>Last Index Fastest</axis_index_order>
<NODE name="Element_Array" />
<FRAGMENT name="Axis_Arrays" />
</Array>"""
)
"""
An interpreted node template to create an ``<Array />``
XML element.
"""

data_2d_contents: NodeBuilderTemplate = interpret_template(
    """<Array_2D_Image>
<offset unit="byte"><NODE name="offset" /></offset>
<axes>2</axes>
<axis_index_order>Last Index Fastest</axis_index_order>
<NODE name="Element_Array" />
<FRAGMENT name="Axis_Arrays" />
</Array_2D_Image>"""
)
"""
An interpreted node template to create an ``<Array_2D_Image />``
XML element.
"""

element_array: NodeBuilderTemplate = interpret_template(
    """<Element_Array>
<data_type><NODE name="data_type" /></data_type></Element_Array>"""
)
"""
An interpreted node template to create an ``<Element_Array />``
XML element.
"""

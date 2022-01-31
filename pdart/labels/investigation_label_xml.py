"""Templates to create a label for Investigation."""
from typing import List

from pdart.labels.namespaces import BUNDLE_NAMESPACES, PDS4_XML_MODEL
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION
from pdart.xml.Templates import (
    DocTemplate,
    NodeBuilderTemplate,
    NodeBuilder,
    FragBuilder,
    interpret_document_template,
    interpret_template,
    combine_nodes_into_fragment,
)

_make_internal_ref_node: NodeBuilderTemplate = interpret_template(
    """<Internal_Reference>
            <lid_reference><NODE name="ref_lid"/></lid_reference>
            <reference_type><NODE name="ref_type"/></reference_type>
      </Internal_Reference>"""
)


def make_internal_ref(ref_lid: str, ref_type: str) -> FragBuilder:
    return _make_internal_ref_node(
        {
            "ref_lid": ref_lid,
            "ref_type": ref_type,
        }
    )


_make_description_node: NodeBuilderTemplate = interpret_template(
    """<description>
<NODE name="description"/>
    </description>"""
)


def make_description(description: str) -> FragBuilder:
    return _make_description_node({"description": description})


make_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
<Product_Context {BUNDLE_NAMESPACES}>
    <Identification_Area>
        <logical_identifier><NODE name="investigation_lid"/></logical_identifier>
        <version_id><NODE name="bundle_vid"/></version_id>
        <title><NODE name="title"/></title>
        <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
        <product_class>Product_Context</product_class>
    <Modification_History>
        <Modification_Detail>
            <modification_date><NODE name="mod_date" /></modification_date>
            <version_id>1.0</version_id>
            <description>Initial set up of the investigation product.</description>
        </Modification_Detail>
    </Modification_History>
    </Identification_Area>
    <Reference_List>
        <FRAGMENT name="internal_reference" />
    </Reference_List>
    <Investigation>
        <name><NODE name="title"/></name>
        <type>Individual Investigation</type>
        <start_date><NODE name="start_date"/></start_date>
        <stop_date><NODE name="stop_date"/></stop_date>
        <FRAGMENT name="description"/>
    </Investigation>
</Product_Context>"""
)
"""
An interpreted document template to create a bundle label.
"""

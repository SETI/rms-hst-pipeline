"""Templates to create a label for a bundle."""
from typing import List

from pdart.labels.Namespaces import BUNDLE_NAMESPACES, PDS4_XML_MODEL
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION
from pdart.xml.Templates import (
    DocTemplate,
    NodeBuilderTemplate,
    NodeBuilder,
    interpret_document_template,
    interpret_template,
    combine_nodes_into_fragment,
)


def make_bundle_context_node(
    time_coordinates_node: NodeBuilder,
    primary_result_summary_node: NodeBuilder,
    investigation_area_node: NodeBuilder,
    target_identification_nodes: List[NodeBuilder],
) -> NodeBuilder:
    func = interpret_template(
        """<Context_Area>
        <NODE name="Time_Coordinates" />
        <NODE name="Primary_Result_Summary" />
        <NODE name="Investigation_Area" />
        <FRAGMENT name="Target_Identification" />
        </Context_Area>"""
    )(
        {
            "Time_Coordinates": time_coordinates_node,
            "Primary_Result_Summary": primary_result_summary_node,
            "Investigation_Area": investigation_area_node,
            "Target_Identification": combine_nodes_into_fragment(
                target_identification_nodes
            ),
        }
    )
    return func


make_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
<Product_Bundle {BUNDLE_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="bundle_lid"/></logical_identifier>
    <version_id><NODE name="bundle_vid"/></version_id>
    <title><NODE name="title"/></title>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Bundle</product_class>
    <NODE name="Citation_Information" />
    <Modification_History>
      <Modification_Detail>
        <modification_date><NODE name="mod_date" /></modification_date>
        <version_id>1.0</version_id>
        <description>Initial PDS4 version</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <FRAGMENT name="Context_Area" />
  <Bundle>
    <bundle_type>Archive</bundle_type>
  </Bundle>
  <FRAGMENT name="Bundle_Member_Entries"/>
</Product_Bundle>"""
)
"""
An interpreted document template to create a bundle label.
"""

make_bundle_entry_member: NodeBuilderTemplate = interpret_template(
    """<Bundle_Member_Entry>
    <lidvid_reference><NODE name="collection_lidvid"/></lidvid_reference>
    <member_status>Primary</member_status>
    <reference_type><NODE name="ref_type"/></reference_type>
</Bundle_Member_Entry>"""
)
"""
An interpreted fragment template to create a ``<Bundle_Member_Entry
/>`` XML element.
"""

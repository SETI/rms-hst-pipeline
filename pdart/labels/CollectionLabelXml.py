"""Templates to create a label for a collection."""
from typing import List

from pdart.labels.Namespaces import COLLECTION_NAMESPACES, PDS4_XML_MODEL
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION
from pdart.xml.Templates import (
    DocTemplate,
    NodeBuilder,
    NodeBuilderTemplate,
    interpret_document_template,
    interpret_template,
    combine_nodes_into_fragment,
)

make_context_collection_title: NodeBuilderTemplate = interpret_template(
    """
    <title>Context collection of <NODE name="instrument"/> observations \
obtained from HST Observing Program <NODE name="proposal_id"/>.</title>
    """
)


make_document_collection_title: NodeBuilderTemplate = interpret_template(
    """
    <title>Document collection of <NODE name="instrument"/> observations \
obtained from HST Observing Program <NODE name="proposal_id"/>.</title>
    """
)


make_schema_collection_title: NodeBuilderTemplate = interpret_template(
    """
    <title>Schema collection of <NODE name="instrument"/> observations \
obtained from HST Observing Program <NODE name="proposal_id"/>.</title>
    """
)


make_other_collection_title: NodeBuilderTemplate = interpret_template(
    """
    <title><NODE name="prefix"/> collection of <NODE name="instrument"/> \
observations obtained from HST Observing Program <NODE name="proposal_id"/>\
.</title>
    """
)


def make_collection_context_node(
    time_coordinates_node: NodeBuilder,
    primary_result_summary_node: NodeBuilder,
    investigation_area_node: NodeBuilder,
    observing_system_node: NodeBuilder,
    target_identification_nodes: List[NodeBuilder],
) -> NodeBuilder:
    func = interpret_template(
        """<Context_Area>
        <NODE name="Time_Coordinates" />
        <NODE name="Primary_Result_Summary" />
        <NODE name="Investigation_Area" />
        <NODE name="Observing_System" />
        <FRAGMENT name="Target_Identification" />
        </Context_Area>"""
    )(
        {
            "Time_Coordinates": time_coordinates_node,
            "Primary_Result_Summary": primary_result_summary_node,
            "Investigation_Area": investigation_area_node,
            "Observing_System": observing_system_node,
            "Target_Identification": combine_nodes_into_fragment(
                target_identification_nodes
            ),
        }
    )
    return func


make_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
<Product_Collection {COLLECTION_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="collection_lid" /></logical_identifier>
    <version_id><NODE name="collection_vid" /></version_id>
    <NODE name="title"/>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Collection</product_class>
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
  <Collection>
    <collection_type><NODE name="collection_type"/></collection_type>
  </Collection>
  <File_Area_Inventory>
    <File>
      <file_name><NODE name="inventory_name" /></file_name>
    </File>
    <Inventory>
      <offset unit="byte">0</offset>
      <parsing_standard_id>PDS DSV 1</parsing_standard_id>
      <records><NODE name="record_count"/></records>
      <record_delimiter>Carriage-Return Line-Feed</record_delimiter>
      <field_delimiter>Comma</field_delimiter>
      <Record_Delimited>
        <fields>2</fields>
        <groups>0</groups>
        <Field_Delimited>
          <name>Member Status</name>
          <field_number>1</field_number>
          <data_type>ASCII_String</data_type>
          <maximum_field_length unit="byte">1</maximum_field_length>
        </Field_Delimited>
        <Field_Delimited>
          <name>LIDVID_LID</name>
          <field_number>2</field_number>
          <data_type>ASCII_LIDVID_LID</data_type>
          <maximum_field_length unit="byte">255</maximum_field_length>
        </Field_Delimited>
      </Record_Delimited>
      <reference_type>inventory_has_member_product</reference_type>
    </Inventory>
  </File_Area_Inventory>
</Product_Collection>"""
)
"""
An interpreted document template to create a collection label.
"""

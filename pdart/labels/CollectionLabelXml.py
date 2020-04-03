"""Templates to create a label for a collection."""
from typing import TYPE_CHECKING

from pdart.labels.Namespaces import COLLECTION_NAMESPACES, PDS4_XML_MODEL
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION, PDS4_SHORT_VERSION
from pdart.xml.Templates import interpret_document_template, interpret_template

from pdart.xml.Templates import DocTemplate, NodeBuilderTemplate


make_document_collection_title = interpret_template(
    """
    <title>This collection contains documentation from \
HST Observing Program <NODE name="proposal_id"/>.</title>
    """
)  # type: NodeBuilderTemplate


make_non_document_collection_title = interpret_template(
    """<title>This collection contains the <NODE name="suffix"/> \
images obtained from HST Observing Program \
<NODE name="proposal_id"/>.</title>"""
)  # type: NodeBuilderTemplate


make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
%s
<Product_Collection %s>
  <Identification_Area>
    <logical_identifier><NODE name="collection_lid" /></logical_identifier>
    <version_id><NODE name="collection_vid" /></version_id>
    <NODE name="title"/>
    <information_model_version>%s</information_model_version>
    <product_class>Product_Collection</product_class>

    <NODE name="Citation_Information" />
  </Identification_Area>
  <Collection>
    <collection_type>Data</collection_type>
  </Collection>
  <File_Area_Inventory>
    <File>
      <file_name><NODE name="inventory_name" /></file_name>
    </File>
    <Inventory>
      <offset unit="byte">0</offset>
      <parsing_standard_id>PDS DSV 1</parsing_standard_id>
      <records>1</records>
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
    % (PDS4_XML_MODEL, COLLECTION_NAMESPACES, INFORMATION_MODEL_VERSION)
)  # type: DocTemplate
"""
An interpreted document template to create a collection label.
"""

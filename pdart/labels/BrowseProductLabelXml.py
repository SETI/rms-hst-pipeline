"""A document template to create a label for a raw browse product."""

from pdart.labels.Namespaces import BROWSE_PRODUCT_NAMESPACES, PDS4_XML_MODEL
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION
from pdart.xml.Templates import DocTemplate, interpret_document_template

make_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
<Product_Browse {BROWSE_PRODUCT_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="browse_lid" /></logical_identifier>
    <version_id><NODE name="browse_vid" /></version_id>
    <title>This product contains a browse image of a <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Browse</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date>2016-04-20</modification_date>
        <version_id>1.0</version_id>
        <description>Initial version</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Reference_List>
    <Internal_Reference>
      <lidvid_reference><NODE name="data_lidvid" /></lidvid_reference>
      <reference_type>browse_to_data</reference_type>
    </Internal_Reference>
  </Reference_List>
  <File_Area_Browse>
    <File>
      <file_name><NODE name="browse_file_name" /></file_name>
    </File>
    <Encoded_Image>
      <offset unit="byte">0</offset>
      <object_length unit="byte"><NODE name="object_length" /></object_length>
      <encoding_standard_id>JPEG</encoding_standard_id>
    </Encoded_Image>
  </File_Area_Browse>
</Product_Browse>"""
)
"""
An interpreted document template to create a label for a RAW browse product.
"""

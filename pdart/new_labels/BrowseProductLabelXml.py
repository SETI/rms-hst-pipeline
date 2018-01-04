"""A document template to create a label for a raw browse product."""
from pdart.xml.Pds4Version import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Browse
   xmlns="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.xsd">
  <Identification_Area>
    <logical_identifier><NODE name="browse_lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title>This product contains a browse image of a <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>%s</information_model_version>
    <product_class>Product_Browse</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date>2016-04-20</modification_date>
        <version_id>0.1</version_id>
        <description>PDS4 version-in-development of the product</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Reference_List>
    <Internal_Reference>
      <lid_reference><NODE name="data_lid" /></lid_reference>
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
</Product_Browse>""" %
    (PDS4_SHORT_VERSION, PDS4_SHORT_VERSION, INFORMATION_MODEL_VERSION))
# type: DocTemplate
"""
An interpreted document template to create a label for a RAW browse product.
"""

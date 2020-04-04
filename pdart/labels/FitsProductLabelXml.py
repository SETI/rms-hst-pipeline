"""Templates to create a label for a product."""

from pdart.labels.Namespaces import (
    FITS_PRODUCT_NAMESPACES,
    HST_XML_MODEL,
    PDS4_XML_MODEL,
)
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION, PDS4_SHORT_VERSION
from pdart.xml.Templates import interpret_document_template

from pdart.xml.Templates import DocTemplate

make_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
{HST_XML_MODEL}
<Product_Observational {FITS_PRODUCT_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="lid" /></logical_identifier>
    <version_id><NODE name="vid" /></version_id>
    <title>This product contains the <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Observational</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date>2016-04-20</modification_date>
        <version_id>0.1</version_id>
        <description>PDS4 version-in-development of the product</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Observation_Area>
    <NODE name="Time_Coordinates" />
    <Investigation_Area>
      <name><NODE name="Investigation_Area_name" /></name>
      <type>Individual Investigation</type>
      <Internal_Reference>
        <lidvid_reference><NODE name="investigation_lidvid" />\
</lidvid_reference>
        <reference_type>data_to_investigation</reference_type>
      </Internal_Reference>
    </Investigation_Area>
    <NODE name="Observing_System" />
    <NODE name="Target_Identification" />
    <Mission_Area><NODE name="HST" /></Mission_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name><NODE name="file_name" /></file_name>
    </File>
    <FRAGMENT name="file_contents" />
  </File_Area_Observational>
</Product_Observational>"""
)
"""
An interpreted document template to create a product label.
"""


def mk_Investigation_Area_name(proposal_id: int) -> str:
    """
    Boilerplate for the text content of a ``<name />`` element in the
    ``<Investigation_Area />`` element.
    """
    return f"HST observing program {proposal_id}"


def mk_Investigation_Area_lidvid(proposal_id: int) -> str:
    """
    Boilerplate for the text content of a ``<lidvid />`` element in
    the ``<Investigation_Area />`` element.
    """
    return f"urn:nasa:pds:context:investigation:investigation.hst_{proposal_id:05}::1.0"

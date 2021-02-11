"""Templates to create a label for a product."""

from pdart.labels.Namespaces import (
    FITS_PRODUCT_NAMESPACES,
    HST_XML_MODEL,
    PDS4_XML_MODEL,
)
from pdart.xml.Pds4Version import INFORMATION_MODEL_VERSION
from pdart.xml.Templates import DocTemplate, interpret_document_template

make_data_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
{HST_XML_MODEL}
<Product_Observational {FITS_PRODUCT_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="lid" /></logical_identifier>
    <version_id><NODE name="vid" /></version_id>
    <title><NODE name="title" /></title>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Observational</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date><NODE name="mod_date" /></modification_date>
        <version_id>1.0</version_id>
        <description>Initial PDS4 version</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <Observation_Area>
    <NODE name="Time_Coordinates" />
    <NODE name="Primary_Result_Summary" />
    <NODE name="Investigation_Area" />
    <NODE name="Observing_System" />
    <FRAGMENT name="Target_Identification" />
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
An interpreted document template to create a data product label.
"""

make_misc_label: DocTemplate = interpret_document_template(
    f"""<?xml version="1.0" encoding="utf-8"?>
{PDS4_XML_MODEL}
{HST_XML_MODEL}
<Product_Ancillary {FITS_PRODUCT_NAMESPACES}>
  <Identification_Area>
    <logical_identifier><NODE name="lid" /></logical_identifier>
    <version_id><NODE name="vid" /></version_id>
    <title><NODE name="title" /></title>
    <information_model_version>{INFORMATION_MODEL_VERSION}</information_model_version>
    <product_class>Product_Ancillary</product_class>
    <Modification_History>
      <Modification_Detail>
        <modification_date><NODE name="mod_date" /></modification_date>
        <version_id>1.0</version_id>
        <description>Initial PDS4 version</description>
      </Modification_Detail>
    </Modification_History>
  </Identification_Area>
  <File_Area_Ancillary>
    <File>
      <file_name><NODE name="file_name" /></file_name>
    </File>
    <FRAGMENT name="file_contents" />
  </File_Area_Ancillary>
</Product_Ancillary>"""
)
"""
An interpreted document template to create a misc product label.
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

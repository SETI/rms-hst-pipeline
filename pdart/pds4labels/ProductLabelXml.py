"""Templates to create a label for a product."""
from pdart.xml.Pds4Version import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Observational
   xmlns="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:hst="http://pds.nasa.gov/pds4/hst/v0"
   xmlns:pds="http://pds.nasa.gov/pds4/pds/v1"
   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
   xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1
                       http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_%s.xsd">
  <Identification_Area>
    <logical_identifier><NODE name="lid" /></logical_identifier>
    <version_id>0.1</version_id>
    <title>This collection contains the <NODE name="suffix" /> \
image obtained the HST Observing Program <NODE name="proposal_id" />\
.</title>
    <information_model_version>%s</information_model_version>
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
</Product_Observational>""" % \
        (PDS4_SHORT_VERSION, PDS4_SHORT_VERSION, PDS4_LONG_VERSION))
"""
An interpreted document template to create a product label.

type: Dict -> Doc
"""


def mk_Investigation_Area_name(proposal_id):
    """
    Boilerplate for the text content of a ``<name />`` element in the
    ``<Investigation_Area />`` element.
    """
    return 'HST observing program %d' % proposal_id


def mk_Investigation_Area_lidvid(proposal_id):
    """
    Boilerplate for the text content of a ``<lidvid />`` element in
    the ``<Investigation_Area />`` element.
    """
    return 'urn:nasa:pds:context:investigation:investigation.hst_%05d::1.0' % \
        proposal_id

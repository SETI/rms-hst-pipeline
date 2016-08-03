from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Bundle xmlns="http://pds.nasa.gov/pds4/pds/v1"
                xmlns:pds="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier><NODE name="lid"/></logical_identifier>
    <version_id>0.1</version_id>
    <title>This bundle contains images obtained from HST Observing Program
<NODE name="proposal_id"/>.</title>
    <information_model_version>1.6.0.0</information_model_version>
    <product_class>Product_Bundle</product_class>
    <NODE name="Citation_Information" />
  </Identification_Area>
  <Bundle>
    <bundle_type>Archive</bundle_type>
  </Bundle>
  <FRAGMENT name="Bundle_Member_Entries"/>
</Product_Bundle>""")

make_bundle_entry_member = interpret_template(
    """<Bundle_Member_Entry>
    <lid_reference><NODE name="lid"/></lid_reference>
    <member_status>Primary</member_status>
    <reference_type>bundle_has_data_collection</reference_type>
</Bundle_Member_Entry>"""
    )

placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year>2000</publication_year>
<description>### placeholder for \
citation_information_description ###</description>
</Citation_Information>""")({})

from pdart.pds4.Collection import *
from pdart.reductions.Reduction import *
from pdart.xml.Templates import *

make_label = interpret_document_template(
    """<?xml version="1.0" encoding="utf-8"?>
<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1500.sch"
            schematypens="http://purl.oclc.org/dsdl/schematron"?>
<Product_Collection xmlns="http://pds.nasa.gov/pds4/pds/v1"
                    xmlns:pds="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier><PARAM name="lid"/></logical_identifier>
    <version_id>0.1</version_id>
    <title>This collection contains the <PARAM name="suffix"/> \
images obtained from HST Observing Program \
<PARAM name="proposal_id"/>.</title>
    <information_model_version>1.5.0.0</information_model_version>
    <product_class>Product_Collection</product_class>

    <PARAM name="Citation_Information" />
  </Identification_Area>
  <Collection>
    <collection_type>Data</collection_type>
  </Collection>
  <File_Area_Inventory>
    <File>
      <file_name><PARAM name="inventory_name" /></file_name>
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
          <name>Member_Status</name>
          <field_number>1</field_number>
          <data_type>ASCII_String</data_type>
          <maximum_field_length unit="byte">1</maximum_field_length>
        </Field_Delimited>
        <Field_Delimited>
          <name>LIDVID_LID</name>
          <field_number>2</field_number>
          <data_type>ASCII_LIDVID_LID</data_type>
        </Field_Delimited>
      </Record_Delimited>
      <reference_type>inventory_has_member_product</reference_type>
    </Inventory>
  </File_Area_Inventory>
</Product_Collection>""")

placeholder_citation_information = interpret_template(
    """<Citation_Information>
<publication_year>2000</publication_year>
<description>### placeholder for \
citation_information_description ###</description>
</Citation_Information>""")({})


class CollectionLabelReduction(Reduction):
    """
    Reduction of a :class:`Collection` to its PDS4 label as a string.
    """
    def reduce_collection(self, archive, lid, get_reduced_products):
        collection = Collection(archive, lid)
        suffix = collection.suffix()
        proposal_id = collection.bundle().proposal_id()
        inventory_name = 'collection_%s_inventory.tab' % suffix

        dict = {'lid': interpret_text(str(lid)),
                'suffix': interpret_text(suffix.upper()),
                'proposal_id': interpret_text(str(proposal_id)),
                'Citation_Information': placeholder_citation_information,
                'inventory_name': interpret_text(inventory_name)
                }
        return make_label(dict).toxml()


def make_collection_label(collection):
    return ReductionRunner().run_collection(CollectionLabelReduction(),
                                            collection)

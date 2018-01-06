import unittest

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.CollectionLabel import *

_COLLECTION_LIDVID = 'urn:nasa:pds:hst_09059:data_acs_raw::1'
_BUNDLE_LIDVID = 'urn:nasa:pds:hst_09059::1.3'


class Test_CollectionLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()
        self.db.create_bundle(_BUNDLE_LIDVID)
        self.db.create_non_document_collection(_COLLECTION_LIDVID,
                                               _BUNDLE_LIDVID)

    @unittest.skip('under construction')
    def test_make_collection_inventory(self):
        # type: () -> None
        inventory = make_collection_inventory(self.db,
                                              _COLLECTION_LIDVID)
        self.assertEqual(_expected_inventory, inventory)

    def test_make_collection_label(self):
        # type: () -> None
        label = make_collection_label(self.db,
                                      _COLLECTION_LIDVID,
                                      True)
        print label
        self.assertEqual(_expected_label, label)


_expected_inventory = u''  # type: unicode

_expected_label = '\n'.join([
    '<?xml version="1.0"?>',
    '<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"',
    '            schematypens="http://purl.oclc.org/dsdl/schematron"?>',
    '<Product_Collection xmlns="http://pds.nasa.gov/pds4/pds/v1" '
    'xmlns:pds="http://pds.nasa.gov/pds4/pds/v1">',
    '  <Identification_Area>',
    '    <logical_identifier>urn:nasa:pds:hst_09059:data_acs_raw'
    '</logical_identifier>',
    '    <version_id>0.1</version_id>',
    '    <title>This collection contains the raw images obtained from HST '
    'Observing Program 9059.</title>',
    '    <information_model_version>1.6.0.0</information_model_version>',
    '    <product_class>Product_Collection</product_class>',
    '    <Citation_Information>',
    '      <publication_year>2000</publication_year>',
    '      <description>### placeholder for Citation_Information/description '
    '###</description>',
    '    </Citation_Information>',
    '  </Identification_Area>',
    '  <Collection>',
    '    <collection_type>Data</collection_type>',
    '  </Collection>',
    '  <File_Area_Inventory>',
    '    <File>',
    '      <file_name>collection.csv</file_name>',
    '    </File>',
    '    <Inventory>',
    '      <offset unit="byte">0</offset>',
    '      <parsing_standard_id>PDS DSV 1</parsing_standard_id>',
    '      <records>1</records>',
    '      <record_delimiter>Carriage-Return Line-Feed</record_delimiter>',
    '      <field_delimiter>Comma</field_delimiter>',
    '      <Record_Delimited>',
    '        <fields>2</fields>',
    '        <groups>0</groups>',
    '        <Field_Delimited>',
    '          <name>Member Status</name>',
    '          <field_number>1</field_number>',
    '          <data_type>ASCII_String</data_type>',
    '          <maximum_field_length unit="byte">1</maximum_field_length>',
    '        </Field_Delimited>',
    '        <Field_Delimited>',
    '          <name>LIDVID_LID</name>',
    '          <field_number>2</field_number>',
    '          <data_type>ASCII_LIDVID_LID</data_type>',
    '          <maximum_field_length unit="byte">255</maximum_field_length>',
    '        </Field_Delimited>',
    '      </Record_Delimited>',
    '      <reference_type>inventory_has_member_product</reference_type>',
    '    </Inventory>',
    '  </File_Area_Inventory>',
    '</Product_Collection>',
    ''
])

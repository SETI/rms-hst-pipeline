import unittest

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.DocumentProductLabel import *
from pdart.xml.Pretty import pretty_print


class Test_DocumentProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_make_document_product_label(self):
        # type: () -> None
        bundle_lidvid = 'urn:nasa:pds:hst_13012::1'
        self.db.create_bundle('urn:nasa:pds:hst_13012::1')

        collection_lidvid = \
            'urn:nasa:pds:hst_13012:document::3.14159'
        self.db.create_document_collection(collection_lidvid,
                                           bundle_lidvid)

        document_product_lidvid = 'urn:nasa:pds:hst_13012:document:phase2::1'

        str = make_document_product_label(self.db,
                                          document_product_lidvid,
                                          True)
        str = pretty_print(str)
        print str
        self.assertEqual(_expected, str)


_expected = '\n'.join([
    '<?xml version="1.0"?>',
    '<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"',
    '            schematypens="http://purl.oclc.org/dsdl/schematron"?>',
    '<Product_Document xmlns="http://pds.nasa.gov/pds4/pds/v1" '
    'xmlns:hst="http://pds.nasa.gov/pds4/hst/v0" '
    'xmlns:pds="http://pds.nasa.gov/pds4/pds/v1" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1                     '
    '        http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">',
    '  <Identification_Area>',
    '    <logical_identifier>urn:nasa:pds:hst_13012:document:phase2'
    '</logical_identifier>',
    '    <version_id>0.1</version_id>',
    '    <title>Summary of the observation plan for HST proposal '
    '13012</title>',
    '    <information_model_version>1.6.0.0</information_model_version>',
    '    <product_class>Product_Document</product_class>',
    '    <Citation_Information>',
    '      <author_list>### placeholder for doc product author_list '
    '###</author_list>',
    '      <publication_year>2000</publication_year>',
    '      <description>This document provides a summary of the observation '
    'plan for HST proposal 13012, ### placeholder for doc product '
    'proposal_title ###, PI ### placeholder for doc product pi_name ###, '
    '2000.</description>',
    '    </Citation_Information>',
    '  </Identification_Area>',
    '  <Reference_List>',
    '    <Internal_Reference>',
    '      <lid_reference>urn:nasa:pds:hst_13012</lid_reference>',
    '      <reference_type>document_to_investigation</reference_type>',
    '    </Internal_Reference>',
    '  </Reference_List>',
    '  <Document>',
    '    <publication_date>2018-01-06</publication_date>',
    '    <Document_Edition>',
    '      <edition_name>0.0</edition_name>',
    '      <language>English</language>',
    '      <files>1</files>',
    '      <Document_File>',
    '        <file_name>phase2.txt</file_name>',
    '        <document_standard_id>7-Bit ASCII Text</document_standard_id>',
    '      </Document_File>',
    '    </Document_Edition>',
    '  </Document>',
    '</Product_Document>',
    ''
])  # type: str
# I'm jumping through hoops here just to get PEP8 compliance.

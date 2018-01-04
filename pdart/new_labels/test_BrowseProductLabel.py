import unittest

from fs.path import basename, join

from pdart.new_db.BrowseFileDB import populate_database_from_browse_file
from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.BrowseProductLabel import *
from pdart.xml.Pretty import pretty_print


class Test_BrowseProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()

    def tearDown(self):
        # type: () -> None
        pass

    def test_make_browse_product_label(self):
        # type: () -> None
        self.db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        bundle_lidvid = \
            'urn:nasa:pds:hst_13012::123'
        self.db.create_bundle(bundle_lidvid)

        fits_collection_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw::3.14159'
        self.db.create_non_document_collection(fits_collection_lidvid,
                                               bundle_lidvid)

        browse_collection_lidvid = \
            'urn:nasa:pds:hst_13012:browse_acs_raw::3.14159'
        self.db.create_non_document_collection(browse_collection_lidvid,
                                               bundle_lidvid)

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2'
        self.db.create_browse_product(fits_product_lidvid,
                                      fits_collection_lidvid)

        browse_product_lidvid = \
            'urn:nasa:pds:hst_13012:browse_acs_raw:jbz504eoq_raw::2'
        self.db.create_browse_product(browse_product_lidvid,
                                      browse_collection_lidvid)

        os_filepath = join(
            archive,
            'hst_13012/browse_acs_raw/visit_04/jbz504eoq_raw.jpg')
        file_basename = basename(os_filepath)

        populate_database_from_browse_file(self.db,
                                           browse_product_lidvid,
                                           browse_collection_lidvid,
                                           file_basename,
                                           5492356)

        browse_file = self.db.get_file(browse_product_lidvid, file_basename)

        # TODO Should I be making from raw ingredients or from the database?
        # And should I have two collections?
        str = make_browse_product_label(self.db,
                                        fits_product_lidvid,
                                        browse_product_lidvid,
                                        file_basename,
                                        True)
        str = pretty_print(str)
        print str
        self.assertEqual(_expected, str)


_expected = '\n'.join([
    '<?xml version="1.0"?>',
    '<?xml-model href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.sch"',
    '            schematypens="http://purl.oclc.org/dsdl/schematron"?>',
    '<Product_Browse xmlns="http://pds.nasa.gov/pds4/pds/v1" '
    'xmlns:pds="http://pds.nasa.gov/pds4/pds/v1" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
    'xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">',
    '  <Identification_Area>',
    '    <logical_identifier>urn:nasa:pds:hst_13012:browse_acs_raw'
    ':jbz504eoq_raw</logical_identifier>',
    '    <version_id>0.1</version_id>',
    '    <title>This product contains a browse image of a raw image obtained '
    'the HST Observing Program 13012.</title>',
    '    <information_model_version>1.6.0.0</information_model_version>',
    '    <product_class>Product_Browse</product_class>',
    '    <Modification_History>',
    '      <Modification_Detail>',
    '        <modification_date>2016-04-20</modification_date>',
    '        <version_id>0.1</version_id>',
    '        <description>PDS4 version-in-development of the '
    'product</description>',
    '      </Modification_Detail>',
    '    </Modification_History>',
    '  </Identification_Area>',
    '  <Reference_List>',
    '    <Internal_Reference>',
    '      <lid_reference>urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw'
    '</lid_reference>',
    '      <reference_type>browse_to_data</reference_type>',
    '    </Internal_Reference>',
    '  </Reference_List>',
    '  <File_Area_Browse>',
    '    <File>',
    '      <file_name>jbz504eoq_raw.jpg</file_name>',
    '    </File>',
    '    <Encoded_Image>',
    '      <offset unit="byte">0</offset>',
    '      <object_length unit="byte">5492356</object_length>',
    '      <encoding_standard_id>JPEG</encoding_standard_id>',
    '    </Encoded_Image>',
    '  </File_Area_Browse>',
    '</Product_Browse>',
    ''])  # type: str
# I'm jumping through hoops here just to get PEP8 compliance.

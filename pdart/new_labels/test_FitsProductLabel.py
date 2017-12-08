import unittest

from fs.path import basename, join

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import card_dictionaries, populate_from_fits_file
from pdart.new_db.SqlAlchTables import File, FitsFile
from pdart.new_labels.FitsProductLabel import *
from pdart.xml.Pretty import pretty_print


class Test_FitsProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()

    def tearDown(self):
        # type: () -> None
        pass

    def test_make_fits_product_label(self):
        # type: () -> None
        self.db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        bundle_lidvid = \
            'urn:nasa:pds:hst_13012::123'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw::3.14159'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2'
        self.db.create_fits_product(fits_product_lidvid,
                                    collection_lidvid)

        os_filepath = join(
            archive,
            'hst_13012/data_acs_raw/visit_04/jbz504eoq_raw.fits')

        populate_from_fits_file(self.db,
                                os_filepath,
                                fits_product_lidvid)

        file_basename = basename(os_filepath)

        fits_file = self.db.session.query(FitsFile).filter(
            File.product_lidvid == fits_product_lidvid).filter(
            File.basename == file_basename).one()

        hdu_count = fits_file.hdu_count

        card_dicts = card_dictionaries(self.db.session,
                                       fits_product_lidvid,
                                       hdu_count)

        str = make_fits_product_label(self.db,
                                      card_dicts,
                                      fits_product_lidvid,
                                      file_basename,
                                      True)
        str = pretty_print(str)
        self.assertEqual(_expected, str)


_expected = '\n'.join(['<?xml version="1.0"?>',
                       '<?xml-model '
                       'href="http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700'
                       '.sch"',
                       '            '
                       'schematypens="http://purl.oclc.org/dsdl/schematron"?>',
                       '<Product_Observational '
                       'xmlns="http://pds.nasa.gov/pds4/pds/v1" '
                       'xmlns:hst="http://pds.nasa.gov/pds4/hst/v0" '
                       'xmlns:pds="http://pds.nasa.gov/pds4/pds/v1" '
                       'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                       'xsi:schemaLocation="http://pds.nasa.gov/pds4/pds/v1   '
                       '                     '
                       'http://pds.nasa.gov/pds4/pds/v1/PDS4_PDS_1700.xsd">',
                       '  <Identification_Area>',
                       '    <logical_identifier>urn:nasa:pds:hst_13012'
                       ':data_acs_raw:jbz504eoq_raw</logical_identifier>',
                       '    <version_id>0.1</version_id>',
                       '    <title>This collection contains the raw image '
                       'obtained the HST Observing Program 13012.</title>',
                       '    '
                       '<information_model_version>1.6.0.0'
                       '</information_model_version>',
                       '    '
                       '<product_class>Product_Observational</product_class>',
                       '    <Modification_History>',
                       '      <Modification_Detail>',
                       '        '
                       '<modification_date>2016-04-20</modification_date>',
                       '        <version_id>0.1</version_id>',
                       '        <description>PDS4 version-in-development of '
                       'the product</description>',
                       '      </Modification_Detail>',
                       '    </Modification_History>',
                       '  </Identification_Area>',
                       '  <Observation_Area>',
                       '    <Time_Coordinates>',
                       '      '
                       '<start_date_time>2012-09-27T20:23:28Z'
                       '</start_date_time>',
                       '      '
                       '<stop_date_time>2012-09-27T20:27:58Z</stop_date_time>',
                       '    </Time_Coordinates>',
                       '    <Investigation_Area>',
                       '      <name>HST observing program 13012</name>',
                       '      <type>Individual Investigation</type>',
                       '      <Internal_Reference>',
                       '        '
                       '<lidvid_reference>urn:nasa:pds:context:investigation'
                       ':investigation.hst_13012::1.0</lidvid_reference>',
                       '        '
                       '<reference_type>data_to_investigation'
                       '</reference_type>',
                       '      </Internal_Reference>',
                       '    </Investigation_Area>',
                       '    <Observing_System>',
                       '      <name>Hubble Space Telescope Advanced Camera '
                       'for Surveys</name>',
                       '      <Observing_System_Component>',
                       '        <name>Hubble Space Telescope</name>',
                       '        <type>Spacecraft</type>',
                       '        <Internal_Reference>',
                       '          '
                       '<lid_reference>urn:nasa:pds:context:instrument_host'
                       ':spacecraft.hst</lid_reference>',
                       '          '
                       '<reference_type>is_instrument_host</reference_type>',
                       '        </Internal_Reference>',
                       '      </Observing_System_Component>',
                       '      <Observing_System_Component>',
                       '        <name>Advanced Camera for Surveys</name>',
                       '        <type>Instrument</type>',
                       '        <Internal_Reference>',
                       '          '
                       '<lid_reference>urn:nasa:pds:context:instrument'
                       ':insthost.acs.acs</lid_reference>',
                       '          '
                       '<reference_type>is_instrument</reference_type>',
                       '        </Internal_Reference>',
                       '      </Observing_System_Component>',
                       '    </Observing_System>',
                       '    <Target_Identification>',
                       '      <name>Uranus</name>',
                       '      <type>Planet</type>',
                       '      <description>The planet Uranus</description>',
                       '      <Internal_Reference>',
                       '        '
                       '<lid_reference>urn:nasa:pds:context:target:uranus'
                       '.planet</lid_reference>',
                       '        '
                       '<reference_type>data_to_target</reference_type>',
                       '      </Internal_Reference>',
                       '    </Target_Identification>',
                       '    <Mission_Area>',
                       '      <hst:HST>',
                       '        <hst:Parameters_General>',
                       '          <hst:stsci_group_id>### placeholder for '
                       'stsci_group_id ###</hst:stsci_group_id>',
                       '          '
                       '<hst:hst_proposal_id>13012</hst:hst_proposal_id>',
                       '          <hst:hst_pi_name>Lamy, Laurent '
                       '</hst:hst_pi_name>',
                       '          '
                       '<hst:hst_target_name>URANUS</hst:hst_target_name>',
                       '          <hst:aperture_type>SBC</hst:aperture_type>',
                       '          '
                       '<hst:exposure_duration>270.0</hst:exposure_duration>',
                       '          '
                       '<hst:exposure_type>NORMAL</hst:exposure_type>',
                       '          '
                       '<hst:filter_name>F165LP+N/A</hst:filter_name>',
                       '          '
                       '<hst:fine_guidance_system_lock_type>FINE</hst'
                       ':fine_guidance_system_lock_type>',
                       '          <hst:gyroscope_mode>### placeholder for '
                       'gyroscope_mode ###</hst:gyroscope_mode>',
                       '          '
                       '<hst:instrument_mode_id>ACCUM</hst'
                       ':instrument_mode_id>',
                       '          '
                       '<hst:moving_target_flag>true</hst:moving_target_flag>',
                       '        </hst:Parameters_General>',
                       '        <hst:Parameters_ACS>',
                       '          <hst:detector_id>SBC</hst:detector_id>',
                       '          <hst:gain_mode_id>### placeholder for '
                       'gain_mode_id ###</hst:gain_mode_id>',
                       '          '
                       '<hst:observation_type>IMAGING</hst:observation_type>',
                       '          '
                       '<hst:repeat_exposure_count>0''</hst'
                       ':repeat_exposure_count>',
                       '          <hst:subarray_flag>### placeholder for '
                       'subarray_flag ###</hst:subarray_flag>',
                       '        </hst:Parameters_ACS>',
                       '      </hst:HST>',
                       '    </Mission_Area>',
                       '  </Observation_Area>',
                       '  <File_Area_Observational>',
                       '    <File>',
                       '      <file_name>jbz504eoq_raw.fits</file_name>',
                       '    </File>',
                       '    <Header>',
                       '      <local_identifier>hdu_0</local_identifier>',
                       '      <offset unit="byte">0</offset>',
                       '      <object_length '
                       'unit="byte">14400</object_length>',
                       '      <parsing_standard_id>FITS '
                       '3.0''</parsing_standard_id>',
                       '      <description>Global FITS Header</description>',
                       '    </Header>',
                       '    <Header>',
                       '      <local_identifier>hdu_1</local_identifier>',
                       '      <offset unit="byte">14400</offset>',
                       '      <object_length unit="byte">8640</object_length>',
                       '      <parsing_standard_id>FITS '
                       '3.0</parsing_standard_id>',
                       '      <description>Global FITS Header</description>',
                       '    </Header>',
                       '    <Array_2D_Image>',
                       '      <offset unit="byte">23040</offset>',
                       '      <axes>2</axes>',
                       '      <axis_index_order>Last Index '
                       'Fastest</axis_index_order>',
                       '      <Element_Array>',
                       '        <data_type>SignedMSB2</data_type>',
                       '      </Element_Array>',
                       '      <Axis_Array>',
                       '        <axis_name>Line</axis_name>',
                       '        <elements>1024</elements>',
                       '        <sequence_number>1</sequence_number>',
                       '      </Axis_Array>',
                       '      <Axis_Array>',
                       '        <axis_name>Sample</axis_name>',
                       '        <elements>1024</elements>',
                       '        <sequence_number>2</sequence_number>',
                       '      </Axis_Array>',
                       '    </Array_2D_Image>',
                       '    <Header>',
                       '      <local_identifier>hdu_2</local_identifier>',
                       '      <offset unit="byte">2122560</offset>',
                       '      <object_length unit="byte">5760</object_length>',
                       '      <parsing_standard_id>FITS '
                       '3.0</parsing_standard_id>',
                       '      <description>Global FITS Header</description>',
                       '    </Header>',
                       '    <Header>',
                       '      <local_identifier>hdu_3</local_identifier>',
                       '      <offset unit="byte">2128320</offset>',
                       '      <object_length unit="byte">5760</object_length>',
                       '      <parsing_standard_id>FITS '
                       '3.0</parsing_standard_id>',
                       '      <description>Global FITS Header</description>',
                       '    </Header>',
                       '  </File_Area_Observational>',
                       '</Product_Observational>',
                       ''])  # type: str
# I'm jumping through hoops here just to get PEP8 compliance.

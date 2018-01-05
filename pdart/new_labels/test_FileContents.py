import unittest

from fs.path import basename, join

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.FileContents import *
from pdart.xml.Pretty import pretty_print


class Test_FileContents(unittest.TestCase):
    def test_get_file_contents(self):
        # type: () -> None
        db = create_bundle_db_in_memory()
        db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2'
        os_filepath = join(
            archive,
            'hst_13012/data_acs_raw/visit_04/jbz504eoq_raw.fits')

        populate_database_from_fits_file(db,
                                         os_filepath,
                                         fits_product_lidvid)

        file_basename = basename(os_filepath)

        card_dicts = db.get_card_dictionaries(fits_product_lidvid,
                                              file_basename)

        fb = get_file_contents(db, card_dicts, fits_product_lidvid)
        doc = _fragment_wrapper({'frag': fb})
        str = doc.toxml()
        str = pretty_print(str)

        expected = """<?xml version="1.0"?>
<Wrapper>
  <Header>
    <local_identifier>hdu_0</local_identifier>
    <offset unit="byte">0</offset>
    <object_length unit="byte">14400</object_length>
    <parsing_standard_id>FITS 3.0</parsing_standard_id>
    <description>Global FITS Header</description>
  </Header>
  <Header>
    <local_identifier>hdu_1</local_identifier>
    <offset unit="byte">14400</offset>
    <object_length unit="byte">8640</object_length>
    <parsing_standard_id>FITS 3.0</parsing_standard_id>
    <description>Global FITS Header</description>
  </Header>
  <Array_2D_Image>
    <offset unit="byte">23040</offset>
    <axes>2</axes>
    <axis_index_order>Last Index Fastest</axis_index_order>
    <Element_Array>
      <data_type>SignedMSB2</data_type>
    </Element_Array>
    <Axis_Array>
      <axis_name>Line</axis_name>
      <elements>1024</elements>
      <sequence_number>1</sequence_number>
    </Axis_Array>
    <Axis_Array>
      <axis_name>Sample</axis_name>
      <elements>1024</elements>
      <sequence_number>2</sequence_number>
    </Axis_Array>
  </Array_2D_Image>
  <Header>
    <local_identifier>hdu_2</local_identifier>
    <offset unit="byte">2122560</offset>
    <object_length unit="byte">5760</object_length>
    <parsing_standard_id>FITS 3.0</parsing_standard_id>
    <description>Global FITS Header</description>
  </Header>
  <Header>
    <local_identifier>hdu_3</local_identifier>
    <offset unit="byte">2128320</offset>
    <object_length unit="byte">5760</object_length>
    <parsing_standard_id>FITS 3.0</parsing_standard_id>
    <description>Global FITS Header</description>
  </Header>
</Wrapper>
"""
        self.assertEquals(expected, str)


_fragment_wrapper = interpret_document_template(
    """<Wrapper><FRAGMENT name="frag" /></Wrapper>""")
# type: DocTemplate

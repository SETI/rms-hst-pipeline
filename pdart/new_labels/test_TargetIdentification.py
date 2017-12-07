import unittest

from fs.path import basename, join

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import card_dictionaries, populate_from_fits_file
from pdart.new_db.SqlAlchTables import File, FitsFile
from pdart.new_labels.TargetIdentification import *
from pdart.xml.Pretty import pretty_print


class Test_TargetIdentification(unittest.TestCase):
    def test_get_default_target(self):
        card_dicts = None
        nb = get_target(card_dicts)
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str = nb(doc).toxml()
        str = pretty_print(str)
        expected = """<?xml version="1.0"?>
<Target_Identification>
  <name>Magrathea</name>
  <type>Planet</type>
  <description>Home of Slartibartfast</description>
  <Internal_Reference>
    <lid_reference>urn:nasa:pds:context:target:magrathea.planet</lid_reference>
    <reference_type>data_to_target</reference_type>
  </Internal_Reference>
</Target_Identification>
"""
        self.assertEqual(expected, str)

    def test_get_target(self):
        # type: () -> None
        db = create_bundle_db_in_memory()
        db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2'
        os_filepath = join(
            archive,
            'hst_13012/data_acs_raw/visit_04/jbz504eoq_raw.fits')

        populate_from_fits_file(db,
                                os_filepath,
                                fits_product_lidvid)

        file_basename = basename(os_filepath)

        fits_file = db.session.query(FitsFile).filter(
            File.product_lidvid == fits_product_lidvid).filter(
            File.basename == file_basename).one()

        hdu_count = fits_file.hdu_count

        card_dicts = card_dictionaries(db.session,
                                       fits_product_lidvid,
                                       hdu_count)

        nb = get_target(card_dicts)
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str = nb(doc).toxml()
        str = pretty_print(str)
        print str
        expected = """<?xml version="1.0"?>
<Target_Identification>
  <name>Uranus</name>
  <type>Planet</type>
  <description>The planet Uranus</description>
  <Internal_Reference>
    <lid_reference>urn:nasa:pds:context:target:uranus.planet</lid_reference>
    <reference_type>data_to_target</reference_type>
  </Internal_Reference>
</Target_Identification>
"""
        self.assertEqual(expected, str)

import unittest
import xml.dom

from fs.path import basename

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.TargetIdentification import get_target
from pdart.new_labels.Utils import path_to_testfile
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

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.5'
        os_filepath = path_to_testfile('jbz504eoq_raw.fits')

        populate_database_from_fits_file(db,
                                         os_filepath,
                                         fits_product_lidvid)

        file_basename = basename(os_filepath)

        card_dicts = db.get_card_dictionaries(fits_product_lidvid,
                                              file_basename)

        nb = get_target(card_dicts)
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str = nb(doc).toxml()
        str = pretty_print(str)
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

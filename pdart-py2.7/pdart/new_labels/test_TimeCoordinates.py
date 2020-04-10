import unittest
import xml.dom

from fs.path import basename

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.TimeCoordinates import get_time_coordinates
from pdart.new_labels.Utils import path_to_testfile
from pdart.xml.Pretty import pretty_print


class Test_TimeCoordinates(unittest.TestCase):
    def test_get_default_target(self):
        card_dicts = None
        nb = get_time_coordinates(None, card_dicts)
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str = nb(doc).toxml()
        str = pretty_print(str)
        expected = """<?xml version="1.0"?>
<Time_Coordinates>
  <start_date_time>2000-01-02Z</start_date_time>
  <stop_date_time>2000-01-02Z</stop_date_time>
</Time_Coordinates>
"""
        self.assertEqual(expected, str)

    def test_get_time_coordinates(self):
        # type: () -> None
        db = create_bundle_db_in_memory()
        db.create_tables()
        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.3'
        os_filepath = path_to_testfile('jbz504eoq_raw.fits')

        populate_database_from_fits_file(db,
                                         os_filepath,
                                         fits_product_lidvid)

        file_basename = basename(os_filepath)

        card_dicts = db.get_card_dictionaries(fits_product_lidvid,
                                              file_basename)

        nb = get_time_coordinates(fits_product_lidvid, card_dicts)
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str = nb(doc).toxml()
        str = pretty_print(str)

        expected = """<?xml version="1.0"?>
<Time_Coordinates>
  <start_date_time>2012-09-27T20:23:28Z</start_date_time>
  <stop_date_time>2012-09-27T20:27:58Z</stop_date_time>
</Time_Coordinates>
"""
        self.assertEqual(expected, str)

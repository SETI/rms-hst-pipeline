import unittest
import xml.dom

from fs.path import basename

from pdart.db.bundle_db import create_bundle_db_in_memory
from pdart.db.fits_file_db import populate_database_from_fits_file
from pdart.labels.lookup import DictLookup
from pdart.labels.time_coordinates import get_start_stop_times, get_time_coordinates
from pdart.labels.utils import path_to_testfile
from pdart.xml.pretty import pretty_print


class TestTimeCoordinates(unittest.TestCase):
    def test_get_default_target(self) -> None:
        fits_product_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.3"
        card_dicts = [
            {"DATE-OBS": "2001-01-02", "TIME-OBS": "08:20:00", "EXPTIME": "1.0"}
        ]
        nb = get_time_coordinates(
            get_start_stop_times(DictLookup("test_get_default_target", card_dicts))
        )
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str: bytes = nb(doc).toxml().encode()
        str = pretty_print(str)
        expected = b"""<?xml version="1.0"?>
<Time_Coordinates>
  <start_date_time>2001-01-02T08:20:00Z</start_date_time>
  <stop_date_time>2001-01-02T08:20:01Z</stop_date_time>
</Time_Coordinates>
"""
        self.assertEqual(expected, str)

    def test_get_time_coordinates(self) -> None:
        db = create_bundle_db_in_memory()
        db.create_tables()
        fits_product_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.3"
        os_filepath = path_to_testfile("jbz504eoq_raw.fits")

        populate_database_from_fits_file(db, os_filepath, fits_product_lidvid)

        file_basename = basename(os_filepath)

        card_dicts = db.get_card_dictionaries(fits_product_lidvid, file_basename)

        nb = get_time_coordinates(
            get_start_stop_times(DictLookup("test_get_time_coordinates", card_dicts))
        )
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str: bytes = nb(doc).toxml().encode()
        str = pretty_print(str)

        expected = b"""<?xml version="1.0"?>
<Time_Coordinates>
  <start_date_time>2012-09-27T20:23:28Z</start_date_time>
  <stop_date_time>2012-09-27T20:27:58Z</stop_date_time>
</Time_Coordinates>
"""
        self.assertEqual(expected, str)

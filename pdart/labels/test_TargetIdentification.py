import unittest
import xml.dom
from typing import Any, Dict, List

from fs.path import basename

from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.labels.Lookup import DictLookup
from pdart.labels.TargetIdentification import get_target, get_target_info
from pdart.labels.Utils import path_to_testfile
from pdart.xml.Pretty import pretty_print


class Test_TargetIdentification(unittest.TestCase):
    def test_get_target(self) -> None:
        db = create_bundle_db_in_memory()
        db.create_tables()

        fits_product_lid = "urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw"
        fits_product_lidvid = f"{fits_product_lid}::2.5"
        os_filepath = path_to_testfile("jbz504eoq_raw.fits")

        populate_database_from_fits_file(db, os_filepath, fits_product_lidvid)

        file_basename = basename(os_filepath)

        card_dicts = db.get_card_dictionaries(fits_product_lidvid, file_basename)

        nb = get_target(get_target_info(DictLookup("test_get_target", card_dicts)))
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str: bytes = nb(doc).toxml().encode()
        str = pretty_print(str)
        expected = b"""<?xml version="1.0"?>
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

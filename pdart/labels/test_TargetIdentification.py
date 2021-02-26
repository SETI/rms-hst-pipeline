import unittest
import xml.dom
from typing import Any, Dict, List, Tuple

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

        # 1. Add record to target identification db
        # 2. Fetch data from target identification db to build xml
        target_id = "13012_1"
        target_identifications: List[Tuple] = [
            ("Uranus", [], "Planet", [], "urn:nasa:pds:context:target:planet.uranus")
        ]
        db.add_record_to_target_identification_db(target_id, target_identifications)
        target_from_db = db.get_target_identifications_based_on_id(target_id)
        target_dict: Dict[str, Any] = {}
        # We only have one entry from db query in this test case.
        target_dict["name"] = target_from_db[0].name
        target_dict["type"] = target_from_db[0].type
        target_dict["alternate_designations"] = target_from_db[0].alternate_designations
        target_dict["description"] = target_from_db[0].description
        target_dict["lid"] = target_from_db[0].lid_reference
        target_dict["reference_type"] = "data_to_target"
        nb = get_target(target_dict)

        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str: bytes = nb(doc).toxml().encode()
        str = pretty_print(str)
        expected = b"""<?xml version="1.0"?>
<Target_Identification>
  <name>Uranus</name>
  <type>Planet</type>
  <Internal_Reference>
    <lid_reference>urn:nasa:pds:context:target:planet.uranus</lid_reference>
    <reference_type>data_to_target</reference_type>
  </Internal_Reference>
</Target_Identification>
"""
        self.assertEqual(expected, str)

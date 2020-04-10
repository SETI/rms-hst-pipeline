import unittest
import xml.dom
from typing import Any, Dict, List

from fs.path import basename

from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.labels.HstParameters import get_hst_parameters
from pdart.labels.Utils import assert_golden_file_equal, path_to_testfile
from pdart.xml.Pretty import pretty_print


class Test_HstParameters(unittest.TestCase):
    # TODO Write cases for other two instruments

    # TODO get_gain_mode_id() is failing on this input file.  Investigate why.

    def test_get_acs_parameters(self) -> None:
        db = create_bundle_db_in_memory()
        db.create_tables()

        fits_product_lidvid = (
            "urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.1976"
        )
        os_filepath = path_to_testfile("jbz504eoq_raw.fits")

        populate_database_from_fits_file(db, os_filepath, fits_product_lidvid)

        file_basename = basename(os_filepath)

        card_dicts = db.get_card_dictionaries(fits_product_lidvid, file_basename)
        shm_card_dicts: List[Dict[str, Any]] = []

        nb = get_hst_parameters(card_dicts, shm_card_dicts, "acs")
        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        str: bytes = nb(doc).toxml().encode()
        str = pretty_print(str)

        assert_golden_file_equal(self, "test_HstParameters.golden.xml", str)

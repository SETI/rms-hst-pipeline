import unittest
import xml.dom
from typing import Any, Dict, List

from fs.path import basename

from pdart.db.bundle_db import create_bundle_db_in_memory
from pdart.db.fits_file_db import populate_database_from_fits_file
from pdart.labels.hst_parameters import get_hst_parameters
from pdart.labels.lookup import CARD_SET, DictLookup, make_hdu_lookups
from pdart.labels.time_coordinates import get_start_stop_times
from pdart.labels.utils import assert_golden_file_equal, path_to_testfile
from pdart.xml.Pretty import pretty_print


class TestHstParameters(unittest.TestCase):
    # TODO Write cases for other two instruments

    # TODO get_gain_mode_id() is failing on this input file.  Investigate why.

    def test_get_acs_parameters(self) -> None:
        db = create_bundle_db_in_memory()
        db.create_tables()

        def get_card_dicts(suffix: str) -> CARD_SET:
            fits_product_lidvid = (
                f"urn:nasa:pds:hst_13012:data_acs_{suffix}:jbz504eoq_{suffix}::1.0"
            )
            os_filepath = path_to_testfile(f"jbz504eoq_{suffix}.fits")
            populate_database_from_fits_file(db, os_filepath, fits_product_lidvid)

            file_basename = basename(os_filepath)
            return db.get_card_dictionaries(fits_product_lidvid, file_basename)

        RAWish_lookups = make_hdu_lookups(
            "test_get_acs_parameters:RAW", get_card_dicts("raw")
        )
        SHFish_lookup = DictLookup("test_get_acs_parameters:SHF", get_card_dicts("spt"))
        nb = get_hst_parameters(RAWish_lookups, SHFish_lookup)

        doc = xml.dom.getDOMImplementation().createDocument(None, None, None)
        text: bytes = nb(doc).toxml().encode()
        text = pretty_print(text)

        assert_golden_file_equal(self, "test_hst_parameters.golden.xml", text)

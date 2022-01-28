from typing import List, Tuple
import os
import os.path
import shutil
import tempfile
import unittest

from fs.path import basename

from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.db.FitsFileDB import populate_database_from_fits_file
from pdart.db.SqlAlchTables import TargetIdentification
from pdart.labels.FitsProductLabel import make_fits_product_label
from pdart.labels.Utils import assert_golden_file_equal, path_to_testfile

from pdart.pipeline.SuffixInfo import get_collection_type  # type: ignore


class Test_FitsProductLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_make_data_fits_product_label(self) -> None:
        bundle_lidvid = "urn:nasa:pds:hst_13012::123.90201"
        self.db.create_bundle(bundle_lidvid)

        # collection_lidvid = "urn:nasa:pds:hst_13012:miscellaneous_acs_spt::3.14159"
        collection_lidvid = "urn:nasa:pds:hst_13012:data_acs_raw::3.14159"
        self.db.create_other_collection(collection_lidvid, bundle_lidvid)

        # Create target identifications db for testing purpose
        target_id = "13012_1"
        target_identifications: List[Tuple] = [
            ("Uranus", [], "Planet", [], "urn:nasa:pds:context:target:planet.uranus")
        ]
        self.db.add_record_to_target_identification_db(
            target_id, target_identifications
        )

        with tempfile.TemporaryDirectory() as working_dir:
            mast_dir = os.path.join(working_dir, "mastDownload")
            os.mkdir(mast_dir)

            def make_lidvid(suffix: str) -> Tuple[str, str]:
                collection_type = get_collection_type(suffix=suffix)
                fits_product_lidvid = f"urn:nasa:pds:hst_13012:{collection_type}_acs_{suffix}:jbz504eoq::1.0"
                self.db.create_fits_product(fits_product_lidvid, collection_lidvid)

                file_basename = f"jbz504eoq_{suffix}.fits"
                os_filepath = path_to_testfile(file_basename)
                temp_filepath = os.path.join(mast_dir, file_basename)
                shutil.copyfile(os_filepath, temp_filepath)
                populate_database_from_fits_file(
                    self.db, temp_filepath, fits_product_lidvid
                )
                return (fits_product_lidvid, file_basename)

            RAWish_product_lidvid, RAWish_file_basename = [
                make_lidvid(suffix) for suffix in ["raw", "spt"]
            ][0]

            label = make_fits_product_label(
                working_dir,
                self.db,
                collection_lidvid,
                RAWish_product_lidvid,
                bundle_lidvid,
                RAWish_file_basename,
                True,
                True,
            )
            print("-----------------------------")
            print(label)
            assert_golden_file_equal(self, "test_FitsProductLabel.golden.xml", label)

    def test_make_misc_fits_product_label(self) -> None:
        bundle_lidvid = "urn:nasa:pds:hst_09059::1.0"
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = "urn:nasa:pds:hst_09059:miscellaneous_acs_spt::1.0"
        self.db.create_other_collection(collection_lidvid, bundle_lidvid)

        # Create target identifications db for testing purpose
        target_id = "9059_1"
        target_identifications: List[Tuple] = [
            (
                "762 Pulcova",
                [],
                "Asteroid",
                [],
                "urn:nasa:pds:context:target:asteroid.762_pulcova",
            )
        ]
        self.db.add_record_to_target_identification_db(
            target_id, target_identifications
        )

        with tempfile.TemporaryDirectory() as working_dir:
            mast_dir = os.path.join(working_dir, "mastDownload")
            os.mkdir(mast_dir)

            def make_lidvid(suffix: str) -> Tuple[str, str]:
                collection_type = get_collection_type(suffix=suffix)
                fits_product_lidvid = f"urn:nasa:pds:hst_09059:{collection_type}_acs_{suffix}:j6gp01mkq::1.0"
                self.db.create_fits_product(fits_product_lidvid, collection_lidvid)

                file_basename = f"j6gp01mkq_{suffix}.fits"
                os_filepath = path_to_testfile(file_basename)
                temp_filepath = os.path.join(mast_dir, file_basename)
                shutil.copyfile(os_filepath, temp_filepath)
                populate_database_from_fits_file(
                    self.db, temp_filepath, fits_product_lidvid
                )
                return (fits_product_lidvid, file_basename)

            RAWish_product_lidvid, RAWish_file_basename = [
                make_lidvid(suffix) for suffix in ["raw", "spt"]
            ][1]

            label = make_fits_product_label(
                working_dir,
                self.db,
                collection_lidvid,
                RAWish_product_lidvid,
                bundle_lidvid,
                RAWish_file_basename,
                True,
                True,
            )
            assert_golden_file_equal(
                self, "test_FitsProductLabel_misc.golden.xml", label
            )

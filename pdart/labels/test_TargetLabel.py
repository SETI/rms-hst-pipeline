import unittest

from typing import List, Tuple

from pdart.db.BundleDB import create_bundle_db_in_memory
from pdart.labels.TargetIdentification import make_context_target_label
from pdart.labels.Utils import assert_golden_file_equal

_BUNDLE_LIDVID = "urn:nasa:pds:hst_15678::1.0"
_TARGET = "asteroid.6478_gault"


class Test_TargetLabel(unittest.TestCase):
    def setUp(self) -> None:
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

        # Create target identifications db for testing purpose
        target_id = "15678_1"
        target_identifications: List[Tuple] = [
            (
                "6478 Gaul",
                [
                    "(6478) Gault",
                    "(6478) 1988 JC1",
                    "1988 JC1",
                    "(6478) 1995 KC1",
                    "1995 KC1",
                    "Gault",
                    "Minor Planet 6478",
                    "NAIF ID 2006478",
                ],
                "Asteroid",
                [],
                "urn:nasa:pds:context:target:asteroid.6478_gault",
            )
        ]
        self.db.add_record_to_target_identification_db(
            target_id, target_identifications
        )

    def test_make_bundle_label(self) -> None:
        label = make_context_target_label(self.db, _TARGET, True, True)
        print(label)
        assert_golden_file_equal(self, "test_TargetLabel.golden.xml", label)

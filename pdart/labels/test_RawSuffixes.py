import unittest

from pdart.labels.RawSuffixes import *


class Test_RawSuffixes(unittest.TestCase):
    def test_associated_lids(self) -> None:
        self.assertEqual(
            [
                LID("urn:nasa:pds:hst_09296:data_acs_raw:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_flt:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_drz:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_crj:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_d0f:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_d0m:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_c0f:j8m3b1blq"),
                LID("urn:nasa:pds:hst_09296:data_acs_c0m:j8m3b1blq"),
            ],
            list(
                associated_lids(
                    LID("urn:nasa:pds:hst_09296:data_acs_asn:j8m3b1010"), "J8M3B1BLQ"
                )
            ),
        )

    def test_associated_lidvids(self) -> None:
        self.assertEqual(
            [
                LIDVID("urn:nasa:pds:hst_09296:data_acs_raw:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_flt:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_drz:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_crj:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_d0f:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_d0m:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_c0f:j8m3b1blq::1.0"),
                LIDVID("urn:nasa:pds:hst_09296:data_acs_c0m:j8m3b1blq::1.0"),
            ],
            list(
                associated_lidvids(
                    LIDVID("urn:nasa:pds:hst_09296:data_acs_asn:j8m3b1010::2.5"),
                    "J8M3B1BLQ",
                )
            ),
        )

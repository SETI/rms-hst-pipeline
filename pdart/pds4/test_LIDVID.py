import unittest
from hypothesis import given
import hypothesis.strategies as st

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pds4.test_LID import lids, lid_strings
from pdart.pds4.test_VID import pdart_vids, pdart_vid_strings


def pdart_lidvid_strings():
    """
    A Hypothesis strategy to generate LIDVID strings where the VID has
    two components.
    """
    return st.builds(
        lambda lid, vid: "%s::%s" % (lid, vid), lid_strings(), pdart_vid_strings()
    )


def pdart_lidvids():
    """
    A Hypothesis strategy to generate LIDVIDs where the VID has two
    components.
    """
    return st.builds(LIDVID, pdart_lidvid_strings())


class TestLIDVID(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(Exception):
            LIDVID("::2.0")
        with self.assertRaises(Exception):
            LIDVID("urn:nasa:pds:ssc01.hirespc.cruise:browse::")
        with self.assertRaises(Exception):
            LIDVID("urn:nasa:pds:ssc01.hirespc.cruise:browse::2.0::3.5")
        with self.assertRaises(Exception):
            LIDVID("urn:nasa:pds:ssc01.hirespc.cruise:browse::2.0.0")

    def test_create_from_lid_and_vid(self):
        lid = LID("urn:nasa:pds:ssc01.hirespc.cruise:browse")
        vid = VID("2.5")
        lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
        self.assertEqual(
            LIDVID("urn:nasa:pds:ssc01.hirespc.cruise:browse::2.5"),
            LIDVID.create_from_lid_and_vid(lid, vid),
        )

    @given(lids(), pdart_vids())
    def test_lidvid_lid_vid_properties(self, lid: LID, vid: VID):
        lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
        self.assertEqual(lidvid.lid(), lid)
        self.assertEqual(lidvid.vid(), vid)

    def test_lid(self):
        self.assertEqual(
            LID("urn:nasa:pds:b:c:p"), LIDVID("urn:nasa:pds:b:c:p::666.666").lid()
        )

    def test_vid(self):
        self.assertEqual(VID("666.0"), LIDVID("urn:nasa:pds:b:c:p::666.0").vid())
        self.assertEqual(VID("3.14159"), LIDVID("urn:nasa:pds:b:c:p::3.14159").vid())

    def test_is_bundle_lidvid(self):
        self.assertTrue(LIDVID("urn:nasa:pds:b::1.0").is_bundle_lidvid())
        self.assertFalse(LIDVID("urn:nasa:pds:b:c::1.0").is_bundle_lidvid())
        self.assertFalse(LIDVID("urn:nasa:pds:b:c:p::1.0").is_bundle_lidvid())

    @given(pdart_lidvids())
    def test_is_bundle_lidvid_property(self, lidvid: LIDVID):
        self.assertEqual(lidvid.is_bundle_lidvid(), lidvid.lid().is_bundle_lid())

    def test_is_collection_lidvid(self):
        self.assertFalse(LIDVID("urn:nasa:pds:b::1.0").is_collection_lidvid())
        self.assertTrue(LIDVID("urn:nasa:pds:b:c::1.0").is_collection_lidvid())
        self.assertFalse(LIDVID("urn:nasa:pds:b:c:p::1.0").is_collection_lidvid())

    @given(pdart_lidvids())
    def test_is_collection_lidvid_property(self, lidvid: LIDVID):
        self.assertEqual(
            lidvid.is_collection_lidvid(), lidvid.lid().is_collection_lid()
        )

    def test_is_product_lidvid(self):
        self.assertFalse(LIDVID("urn:nasa:pds:b::1.0").is_product_lidvid())
        self.assertFalse(LIDVID("urn:nasa:pds:b:c::1.0").is_product_lidvid())
        self.assertTrue(LIDVID("urn:nasa:pds:b:c:p::1.0").is_product_lidvid())

    @given(pdart_lidvids())
    def test_is_product_lidvid_property(self, lidvid: LIDVID):
        self.assertEqual(lidvid.is_product_lidvid(), lidvid.lid().is_product_lid())

    def test_next_major_lidvid(self):
        self.assertEqual(
            LIDVID("urn:nasa:pds:b:c:p::667.0"),
            LIDVID("urn:nasa:pds:b:c:p::666.0").next_major_lidvid(),
        )
        self.assertEqual(
            LIDVID("urn:nasa:pds:b:c:p::4.0"),
            LIDVID("urn:nasa:pds:b:c:p::3.14159").next_major_lidvid(),
        )

    @given(pdart_lidvids())
    def test_is_next_major_lidvid_property(self, lidvid: LIDVID):
        self.assertEqual(
            LIDVID.create_from_lid_and_vid(lidvid.lid(), lidvid.vid().next_major_vid()),
            lidvid.next_major_lidvid(),
        )

    def test_next_minor_lidvid(self):
        self.assertEqual(
            LIDVID("urn:nasa:pds:b:c:p::666.10"),
            LIDVID("urn:nasa:pds:b:c:p::666.9").next_minor_lidvid(),
        )
        self.assertEqual(
            LIDVID("urn:nasa:pds:b:c:p::3.14160"),
            LIDVID("urn:nasa:pds:b:c:p::3.14159").next_minor_lidvid(),
        )

    @given(pdart_lidvids())
    def test_is_next_minor_lidvid_property(self, lidvid: LIDVID):
        self.assertEqual(
            LIDVID.create_from_lid_and_vid(lidvid.lid(), lidvid.vid().next_minor_vid()),
            lidvid.next_minor_lidvid(),
        )

    def test_eq(self):
        self.assertTrue(
            LIDVID("urn:nasa:pds:b:c:p::1.0") == LIDVID("urn:nasa:pds:b:c:p::1.0")
        )
        self.assertFalse(
            LIDVID("urn:nasa:pds:b:c:p::1.1") == LIDVID("urn:nasa:pds:b:c:p::1.0")
        )
        self.assertTrue(
            LIDVID("urn:nasa:pds:b:c:p::1.1") != LIDVID("urn:nasa:pds:b:c:p::1.0")
        )
        self.assertFalse(
            LIDVID("urn:nasa:pds:b:c:p::1.0") != LIDVID("urn:nasa:pds:b:c:p::1.0")
        )

    @given(pdart_lidvid_strings(), pdart_lidvid_strings())
    def test_eq_property(self, lhs: str, rhs: str):
        # two LIDVIDs are equal iff their strings are equal
        self.assertEqual(lhs == rhs, LIDVID(lhs) == LIDVID(rhs))

    def test_str(self):
        self.assertEqual(
            "urn:nasa:pds:b:c:p::1.0", str(LIDVID("urn:nasa:pds:b:c:p::1.0"))
        )

    @given(pdart_lidvid_strings())
    def test_str_roundtrip_property(self, lidvid_str):
        """
        Creating a LIDVID from a string and turning it back into a
        string should result in the same string.
        """
        self.assertEqual(lidvid_str, str(LIDVID(lidvid_str)))

    def test_repr(self):
        self.assertEqual(
            "LIDVID('urn:nasa:pds:b:c:p::1.0')", repr(LIDVID("urn:nasa:pds:b:c:p::1.0"))
        )

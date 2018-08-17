import unittest

from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID


class TestLIDVID(unittest.TestCase):
    def test_init(self):
        # type: () -> None
        # sanity-check
        with self.assertRaises(Exception):
            LIDVID(None)

        with self.assertRaises(Exception):
            LIDVID('::2.0')
        with self.assertRaises(Exception):
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::')
        with self.assertRaises(Exception):
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::2.0::3.5')
        with self.assertRaises(Exception):
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::2.0.0')

    def test_create_from_lid_and_vid(self):
        # type: () -> None
        lid = LID('urn:nasa:pds:ssc01.hirespc.cruise:browse')
        vid = VID('2.5')
        lidvid = LIDVID.create_from_lid_and_vid(lid, vid)
        self.assertEqual(
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::2.5'),
            LIDVID.create_from_lid_and_vid(lid, vid))

    def test_lid(self):
        self.assertEqual(LID('urn:nasa:pds:b:c:p'),
                         LIDVID('urn:nasa:pds:b:c:p::666.666').lid())

    def test_vid(self):
        self.assertEqual(VID('666.0'),
                         LIDVID('urn:nasa:pds:b:c:p::666.0').vid())
        self.assertEqual(VID('3.14159'),
                         LIDVID('urn:nasa:pds:b:c:p::3.14159').vid())

    def test_is_bundle_lidvid(self):
        self.assertTrue(LIDVID('urn:nasa:pds:b::1.0').is_bundle_lidvid())
        self.assertFalse(LIDVID('urn:nasa:pds:b:c::1.0').is_bundle_lidvid())
        self.assertFalse(LIDVID('urn:nasa:pds:b:c:p::1.0').is_bundle_lidvid())

    def test_is_collection_lidvid(self):
        self.assertFalse(LIDVID('urn:nasa:pds:b::1.0').is_collection_lidvid())
        self.assertTrue(LIDVID('urn:nasa:pds:b:c::1.0').is_collection_lidvid())
        self.assertFalse(
            LIDVID('urn:nasa:pds:b:c:p::1.0').is_collection_lidvid())

    def test_is_product_lidvid(self):
        self.assertFalse(LIDVID('urn:nasa:pds:b::1.0').is_product_lidvid())
        self.assertFalse(LIDVID('urn:nasa:pds:b:c::1.0').is_product_lidvid())
        self.assertTrue(LIDVID('urn:nasa:pds:b:c:p::1.0').is_product_lidvid())

    def test_next_major_lidvid(self):
        self.assertEqual(
            LIDVID('urn:nasa:pds:b:c:p::667.0'),
            LIDVID('urn:nasa:pds:b:c:p::666.0').next_major_lidvid())
        self.assertEqual(
            LIDVID('urn:nasa:pds:b:c:p::4.0'),
            LIDVID('urn:nasa:pds:b:c:p::3.14159').next_major_lidvid())

    def test_next_minor_lidvid(self):
        self.assertEqual(
            LIDVID('urn:nasa:pds:b:c:p::666.10'),
            LIDVID('urn:nasa:pds:b:c:p::666.9').next_minor_lidvid())
        self.assertEqual(
            LIDVID('urn:nasa:pds:b:c:p::3.14160'),
            LIDVID('urn:nasa:pds:b:c:p::3.14159').next_minor_lidvid())

    def test_eq(self):
        # type: () -> None
        self.assertTrue(LIDVID('urn:nasa:pds:b:c:p::1.0') ==
                        LIDVID('urn:nasa:pds:b:c:p::1.0'))
        self.assertFalse(LIDVID('urn:nasa:pds:b:c:p::1.1') ==
                         LIDVID('urn:nasa:pds:b:c:p::1.0'))
        self.assertTrue(LIDVID('urn:nasa:pds:b:c:p::1.1') !=
                        LIDVID('urn:nasa:pds:b:c:p::1.0'))
        self.assertFalse(LIDVID('urn:nasa:pds:b:c:p::1.0') !=
                         LIDVID('urn:nasa:pds:b:c:p::1.0'))

    def test_str(self):
        # type: () -> None
        self.assertEquals('urn:nasa:pds:b:c:p::1.0',
                          str(LIDVID('urn:nasa:pds:b:c:p::1.0')))

    def test_repr(self):
        # type: () -> None
        self.assertEquals("LIDVID('urn:nasa:pds:b:c:p::1.0')",
                          repr(LIDVID('urn:nasa:pds:b:c:p::1.0')))


if __name__ == '__main__':
    unittest.main()

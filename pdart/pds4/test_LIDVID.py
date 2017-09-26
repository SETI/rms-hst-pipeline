import unittest

from pdart.pds4.LIDVID import *


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

    def test_lid(self):
        self.assertEqual(LID('urn:nasa:pds:b:c:p'),
                         LIDVID('urn:nasa:pds:b:c:p::666').lid())

    def test_vid(self):
        self.assertEqual(VID('666'),
                         LIDVID('urn:nasa:pds:b:c:p::666').vid())
        self.assertEqual(VID('3.14159'),
                         LIDVID('urn:nasa:pds:b:c:p::3.14159').vid())

    def test_next_major_lidvid(self):
        self.assertEqual(LIDVID('urn:nasa:pds:b:c:p::667'),
                         LIDVID('urn:nasa:pds:b:c:p::666').next_major_lidvid())
        self.assertEqual(
            LIDVID('urn:nasa:pds:b:c:p::4'),
            LIDVID('urn:nasa:pds:b:c:p::3.14159').next_major_lidvid())

    def test_next_minor_lidvid(self):
        self.assertEqual(LIDVID('urn:nasa:pds:b:c:p::666.1'),
                         LIDVID('urn:nasa:pds:b:c:p::666').next_minor_lidvid())
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

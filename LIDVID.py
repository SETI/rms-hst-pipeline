import LID
import VID
import unittest


class LIDVID(object):
    """Representation of a PDS4 LIDVID."""

    def __init__(self, str):
        """
        Create a LIDVID object from a string, throwing an exception if
        the LIDVID string is malformed.
        """
        segs = str.split('::')
        assert len(segs) == 2
        self.lidvid = str
        self.lid = LID.LID(segs[0])
        self.vid = VID.VID(segs[1])

    def __eq__(self, other):
        return (self.lid == other.lid) and (self.vid == other.vid)

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return self.lidvid

    def __repr__(self):
        return 'LIDVID(%r)' % self.lidvid

############################################################


class TestLIDVID(unittest.TestCase):
    def test_init(self):
        # sanity-check
        with self.assertRaises(Exception):
            LIDVID(null)

        with self.assertRaises(Exception):
            LIDVID('::2.0')
        with self.assertRaises(Exception):
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::')
        with self.assertRaises(Exception):
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::2.0::3.5')
        with self.assertRaises(Exception):
            LIDVID('urn:nasa:pds:ssc01.hirespc.cruise:browse::2.0.0')

    def test_eq(self):
        self.assertTrue(LIDVID('urn:nasa:pds:b:c:p::1.0') ==
                        LIDVID('urn:nasa:pds:b:c:p::1.0'))
        self.assertFalse(LIDVID('urn:nasa:pds:b:c:p::1.1') ==
                         LIDVID('urn:nasa:pds:b:c:p::1.0'))
        self.assertTrue(LIDVID('urn:nasa:pds:b:c:p::1.1') !=
                        LIDVID('urn:nasa:pds:b:c:p::1.0'))
        self.assertFalse(LIDVID('urn:nasa:pds:b:c:p::1.0') !=
                         LIDVID('urn:nasa:pds:b:c:p::1.0'))

    def test_str(self):
        self.assertEquals('urn:nasa:pds:b:c:p::1.0',
                          str(LIDVID('urn:nasa:pds:b:c:p::1.0')))

    def test_repr(self):
        self.assertEquals("LIDVID('urn:nasa:pds:b:c:p::1.0')",
                          repr(LIDVID('urn:nasa:pds:b:c:p::1.0')))

if __name__ == '__main__':
    unittest.main()

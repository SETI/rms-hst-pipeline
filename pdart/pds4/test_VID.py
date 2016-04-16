import unittest

from pdart.pds4.VID import *


class TestVID(unittest.TestCase):
    def test_init(self):
        # sanity-check
        with self.assertRaises(Exception):
            VID(None)

        with self.assertRaises(Exception):
            VID('foo')

        VID('0.0')
        with self.assertRaises(Exception):
            VID('0.0.0')
        with self.assertRaises(Exception):
            VID('5.')
        with self.assertRaises(Exception):
            VID('.5')
        with self.assertRaises(Exception):
            VID('0.01')

        # test fields
        v = VID('3.14159265')
        self.assertEqual(3, v.major)
        self.assertEqual(14159265, v.minor)

    def test_cmp(self):
        self.assertTrue(VID('2.3') == VID('2.3'))
        self.assertTrue(VID('2.3') != VID('2.4'))
        self.assertTrue(VID('2.3') < VID('3.2'))
        self.assertTrue(VID('2.3') > VID('2.2'))

    def test_str(self):
        self.assertEquals('2.3', str(VID('2.3')))

    def test_repr(self):
        self.assertEquals("VID('2.3')", repr(VID('2.3')))

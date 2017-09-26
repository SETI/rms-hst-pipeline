import unittest

from pdart.pds4.VID import *


class TestVID(unittest.TestCase):
    def test_init(self):
        # type: () -> None
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
        self.assertEqual(3, v._major)
        self.assertEqual(14159265, v._minor)

    def test_next_major_vid(self):
        # type: () -> None
        self.assertEqual(VID('3'), VID('2.9').next_major_vid())

    def test_next_minor_vid(self):
        # type: () -> None
        self.assertEquals(VID('2.1'), VID('2').next_minor_vid())
        self.assertEquals(VID('2.10'), VID('2.9').next_minor_vid())

    def test_cmp(self):
        # type: () -> None
        self.assertTrue(VID('2.3') == VID('2.3'))
        self.assertTrue(VID('2.3') != VID('2.4'))
        self.assertTrue(VID('2.3') < VID('3.2'))
        self.assertTrue(VID('2.3') > VID('2.2'))

    def test_str(self):
        # type: () -> None
        self.assertEquals('2.3', str(VID('2.3')))

    def test_repr(self):
        # type: () -> None
        self.assertEquals("VID('2.3')", repr(VID('2.3')))

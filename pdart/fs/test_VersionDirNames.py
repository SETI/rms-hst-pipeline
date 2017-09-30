import unittest

from VersionDirNames import *


class TestVersionDirNames(unittest.TestCase):
    def test_is_dir_name(self):
        self.assertFalse(is_dir_name(u'xxx'))
        self.assertFalse(is_dir_name(u'v$x'))
        self.assertTrue(is_dir_name(u'v$0'))
        self.assertFalse(is_dir_name(u'v$01'))
        self.assertTrue(is_dir_name(u'v$1'))
        self.assertTrue(is_dir_name(u'v$314159265'))
        self.assertFalse(is_dir_name(u'v$314159265x'))
        self.assertTrue(is_dir_name(u'v$1.2'))
        self.assertFalse(is_dir_name(u'v$1..2'))
        self.assertFalse(is_dir_name(u'v$1.02'))
        self.assertTrue(is_dir_name(u'v$3.14159265'))

    def test_vid_to_dir_name(self):
        self.assertEqual(u'v$1', vid_to_dir_name(VID(u'1')))
        self.assertEqual(u'v$1.5', vid_to_dir_name(VID(u'1.5')))

    def test_dir_name_to_vid(self):
        self.assertEqual(VID(u'3'), dir_name_to_vid(u'v$3'))
        self.assertEqual(VID(u'3.4'), dir_name_to_vid(u'v$3.4'))
        with self.assertRaises(Exception):
            dir_name_to_vid(u'kaboom')
        with self.assertRaises(Exception):
            dir_name_to_vid(u'v$1.2.3')

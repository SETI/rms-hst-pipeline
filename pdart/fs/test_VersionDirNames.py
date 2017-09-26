import unittest

from VersionDirNames import *


class TestVersionDirNames(unittest.TestCase):
    def test_vid_to_dir_name(self):
        self.assertEqual('v$1', vid_to_dir_name(VID('1')))
        self.assertEqual('v$1.5', vid_to_dir_name(VID('1.5')))

    def test_dir_name_to_vid(self):
        self.assertEqual(VID('3'), dir_name_to_vid(u'v$3'))
        self.assertEqual(VID('3.4'), dir_name_to_vid(u'v$3.4'))
        with self.assertRaises(Exception):
            dir_name_to_vid('kaboom')

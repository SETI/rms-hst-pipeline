import unittest

from pdart.fs.LidToDirName import *


class Test_LidToDirName(unittest.TestCase):
    def test_lid_to_dir_name(self):
        # type: () -> None
        self.assertEqual(u'/b', lid_to_dir_name(LID('urn:nasa:pds:b')))
        self.assertEqual(u'/b/c', lid_to_dir_name(LID('urn:nasa:pds:b:c')))
        self.assertEqual(u'/b/c/p', lid_to_dir_name(LID('urn:nasa:pds:b:c:p')))

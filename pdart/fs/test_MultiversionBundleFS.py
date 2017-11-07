import unittest

from fs.memoryfs import MemoryFS

from pdart.fs.MultiversionBundleFS import *
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID


class TestMultiversionBundleFS(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.fs = MultiversionBundleFS(MemoryFS())

    def test_make_lidvid_directories(self):
        self.fs.make_lidvid_directories(LIDVID('urn:nasa:pds:b::3'))
        self.assertTrue(self.fs.exists(u'/b/v$3'))
        self.assertTrue(self.fs.exists(u'/b/v$3/subdir$versions.txt'))
        # arbitrarily set the text (to something illegal)
        self.fs.settext(u'/b/v$3/subdir$versions.txt', u'placeholder')
        # try rerunning
        self.fs.make_lidvid_directories(LIDVID('urn:nasa:pds:b::3'))
        # make sure we didn't overwrite
        self.assertEqual(u'placeholder',
                         self.fs.gettext(u'/b/v$3/subdir$versions.txt'))

        self.fs.make_lidvid_directories(LIDVID('urn:nasa:pds:b:c:p::1'))
        # make sure we didn't create a subdir$versions file, since products
        # don't have subdirs.
        self.assertTrue(self.fs.exists(u'/b/c/p/v$1'))
        self.assertFalse(self.fs.exists(u'/b/c/p/v$1/subdir$versions.txt'))

    def test_current_vid(self):
        # type: () -> None
        lid = LID('urn:nasa:pds:b')
        self.fs.make_lid_directories(lid)
        self.assertEqual(VID('0'), self.fs.current_vid(lid))

        # I have tests for the functions but need tests for the class
        # MultiversionBundleFS itself.

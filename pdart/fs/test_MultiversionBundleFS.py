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

    def test_lidvid_to_subdir_versions_path(self):
        # type: () -> None
        self.assertEqual(u'/b/v$3/subdir$versions.txt',
                         lidvid_to_subdir_versions_path(
                             LIDVID('urn:nasa:pds:b::3')))
        self.assertEqual(u'/b/c/v$4/subdir$versions.txt',
                         lidvid_to_subdir_versions_path(
                             LIDVID('urn:nasa:pds:b:c::4')))
        self.assertEqual(u'/b/c/p/v$5/subdir$versions.txt',
                         lidvid_to_subdir_versions_path(
                             LIDVID('urn:nasa:pds:b:c:p::5')))

    def test_current_vid(self):
        # type: () -> None
        lid = LID('urn:nasa:pds:b')
        self.fs.make_lid_directories(lid)
        self.assertEqual(VID('0'), self.fs.current_vid(lid))

    # I have tests for the functions but need tests for the class
    # MultiversionBundleFS itself.

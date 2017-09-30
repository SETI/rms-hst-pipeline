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

    def test_lid_to_versions_directory_path(self):
        # type: () -> None
        self.assertEqual(u'/b',
                         lid_to_versions_directory_path(
                             LID('urn:nasa:pds:b')))
        self.assertEqual(u'/b/c',
                         lid_to_versions_directory_path(
                             LID('urn:nasa:pds:b:c')))
        self.assertEqual(u'/b/c/p',
                         lid_to_versions_directory_path(
                             LID('urn:nasa:pds:b:c:p')))

    def test_lidvid_to_contents_directory_path(self):
        # type: () -> None
        self.assertEqual(u'/b/v$1',
                         lidvid_to_contents_directory_path(
                             LIDVID('urn:nasa:pds:b::1')))
        self.assertEqual(u'/b/c/v$2',
                         lidvid_to_contents_directory_path(
                             LIDVID('urn:nasa:pds:b:c::2')))
        self.assertEqual(u'/b/c/p/v$3',
                         lidvid_to_contents_directory_path(
                             LIDVID('urn:nasa:pds:b:c:p::3')))

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

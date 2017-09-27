import unittest

from pdart.fs.MultiversionBundleFS import *
from pdart.pds4.LIDVID import LIDVID


class TestMultiversionBundleFS(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.fs = MultiversionBundleFS()

    def test_lidvid_to_files_directory(self):
        # type: () -> None
        self.assertEqual(u'/b/v$1',
                         self.fs.lidvid_to_files_directory(
                             LIDVID('urn:nasa:pds:b::1')))
        self.assertEqual(u'/b/c/v$2',
                         self.fs.lidvid_to_files_directory(
                             LIDVID('urn:nasa:pds:b:c::2')))
        self.assertEqual(u'/b/c/p/v$3',
                         self.fs.lidvid_to_files_directory(
                             LIDVID('urn:nasa:pds:b:c:p::3')))

    def test_lidvid_to_subdir_versions_path(self):
        # type: () -> None
        self.assertEqual(u'/b/v$3/subdir$versions.txt',
                         self.fs.lidvid_to_subdir_versions_path(
                             LIDVID('urn:nasa:pds:b::3')))
        self.assertEqual(u'/b/c/v$4/subdir$versions.txt',
                         self.fs.lidvid_to_subdir_versions_path(
                             LIDVID('urn:nasa:pds:b:c::4')))
        self.assertEqual(u'/b/c/p/v$5/subdir$versions.txt',
                         self.fs.lidvid_to_subdir_versions_path(
                             LIDVID('urn:nasa:pds:b:c:p::5')))

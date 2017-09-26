import unittest

from MultiversionBundleFS import *
from pdart.pds4.LIDVID import LIDVID


class TestMultiversionBundleFS(unittest.TestCase):
    def setUp(self):
        self.fs = MultiversionBundleFS()

    def test_lidvid_to_directory(self):
        self.assertEqual(u'/b/v$1', self.fs.lidvid_to_directory(LIDVID('urn:nasa:pds:b::1')))
        self.assertEqual(u'/b/c/v$2', self.fs.lidvid_to_directory(LIDVID('urn:nasa:pds:b:c::2')))
        self.assertEqual(u'/b/c/p/v$3', self.fs.lidvid_to_directory(LIDVID('urn:nasa:pds:b:c:p::3')))

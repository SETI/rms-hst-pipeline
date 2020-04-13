import unittest

from fs.memoryfs import MemoryFS

from pdart.fs.SubdirVersions import (
    parse_subdir_versions,
    read_subdir_versions_from_directory,
    str_subdir_versions,
    write_subdir_versions_to_directory,
)
from pdart.fs.VersionedFS import ROOT, SUBDIR_VERSIONS_FILENAME


class TestSubdirVersions(unittest.TestCase):
    def setUp(self):
        self.d = {u"baz": u"666.666", u"foo": u"1", u"bar": u"2"}
        self.txt = u"""bar 2
baz 666.666
foo 1
"""

    def testParseSubdirVersions(self):
        self.assertEqual(self.d, parse_subdir_versions(self.txt))
        self.assertEqual({}, parse_subdir_versions(u""))

    def testStrSubdirVersions(self):
        self.assertEqual(self.txt, str_subdir_versions(self.d))
        self.assertEqual(u"", str_subdir_versions({}))

    def testWriteSubdirVersions(self):
        fs = MemoryFS()
        write_subdir_versions_to_directory(fs, ROOT, self.d)
        self.assertEqual(self.txt, fs.gettext(SUBDIR_VERSIONS_FILENAME))
        fs.close()

    def testReadSubdirVersions(self):
        fs = MemoryFS()
        fs.settext(SUBDIR_VERSIONS_FILENAME, self.txt)
        d = read_subdir_versions_from_directory(fs, ROOT)
        self.assertEqual(self.d, d)
        fs.close()
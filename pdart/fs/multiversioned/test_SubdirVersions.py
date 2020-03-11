import unittest

from fs.memoryfs import MemoryFS

from pdart.fs.multiversioned.SubdirVersions import (
    SUBDIR_VERSIONS_FILENAME,
    parse_subdir_versions,
    read_subdir_versions_from_directory,
    unparse_subdir_versions,
    write_subdir_versions_to_directory,
)

ROOT = u"/"  # type: unicode


class Test_SubdirVersions(unittest.TestCase):
    def setUp(self):
        self.d = {u"baz": u"666.666", u"foo": u"1", u"bar": u"2"}
        self.txt = u"""bar 2
baz 666.666
foo 1
"""

    def test_parse_subdir_versions(self):
        self.assertEqual(self.d, parse_subdir_versions(self.txt))
        self.assertEqual({}, parse_subdir_versions(u""))

    def test_unparse_subdir_versions(self):
        self.assertEqual(self.txt, unparse_subdir_versions(self.d))
        self.assertEqual(u"", unparse_subdir_versions({}))

    def test_write_subdir_versions(self):
        fs = MemoryFS()
        write_subdir_versions_to_directory(fs, ROOT, self.d)
        self.assertEqual(self.txt, fs.readtext(SUBDIR_VERSIONS_FILENAME))
        fs.close()

    def test_read_subdir_versions(self):
        fs = MemoryFS()
        d = read_subdir_versions_from_directory(fs, ROOT)
        self.assertEqual({}, d)
        fs.writetext(SUBDIR_VERSIONS_FILENAME, self.txt)
        d = read_subdir_versions_from_directory(fs, ROOT)
        self.assertEqual(self.d, d)
        fs.close()

import unittest

from fs.memoryfs import MemoryFS

from pdart.fs.multiversioned.SubdirVersions import (
    SUBDIR_VERSIONS_FILENAME,
    parse_subdir_versions,
    read_subdir_versions_from_directory,
    unparse_subdir_versions,
    write_subdir_versions_to_directory,
)


class Test_SubdirVersions(unittest.TestCase):
    def setUp(self) -> None:
        self.d = {"baz": "666.666", "foo": "1", "bar": "2"}
        self.txt = """bar 2
baz 666.666
foo 1
"""

    def test_parse_subdir_versions(self) -> None:
        self.assertEqual(self.d, parse_subdir_versions(self.txt))
        self.assertEqual({}, parse_subdir_versions(""))

    def test_unparse_subdir_versions(self) -> None:
        self.assertEqual(self.txt, unparse_subdir_versions(self.d))
        self.assertEqual("", unparse_subdir_versions({}))

    def test_write_subdir_versions(self) -> None:
        fs = MemoryFS()
        write_subdir_versions_to_directory(fs, "/", self.d)
        self.assertEqual(self.txt, fs.readtext(SUBDIR_VERSIONS_FILENAME))
        fs.close()

    def test_read_subdir_versions(self) -> None:
        fs = MemoryFS()
        d = read_subdir_versions_from_directory(fs, "/")
        self.assertEqual({}, d)
        fs.writetext(SUBDIR_VERSIONS_FILENAME, self.txt)
        d = read_subdir_versions_from_directory(fs, "/")
        self.assertEqual(self.d, d)
        fs.close()

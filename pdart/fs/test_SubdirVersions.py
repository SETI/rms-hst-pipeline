import unittest
from fs.memoryfs import MemoryFS

from pdart.fs.SubdirVersions import *
from pdart.fs.VersionedFS import ROOT, SUBDIR_VERSIONS_FILENAME


class TestSubdirVersions(unittest.TestCase):
    def setUp(self):
        self.d = {u'baz': u'666.666', u'foo': u'1', u'bar': u'2'}
        self.txt = u"""bar 2
baz 666.666
foo 1
"""

    def testParseSubdirVersions(self):
        self.assertEqual(self.d, parseSubdirVersions(self.txt))
        self.assertEqual({}, parseSubdirVersions(u''))

    def testStrSubdirVersions(self):
        self.assertEqual(self.txt, strSubdirVersions(self.d))
        self.assertEqual(u'', strSubdirVersions({}))

    def testWriteSubdirVersions(self):
        fs = MemoryFS()
        writeSubdirVersions(fs, ROOT, self.d)
        self.assertEqual(self.txt, fs.gettext(SUBDIR_VERSIONS_FILENAME))
        fs.close()

    def testReadSubdirVersions(self):
        fs = MemoryFS()
        fs.settext(SUBDIR_VERSIONS_FILENAME, self.txt)
        d = readSubdirVersions(fs, ROOT)
        self.assertEqual(self.d, d)
        fs.close()

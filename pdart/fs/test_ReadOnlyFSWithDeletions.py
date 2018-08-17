import unittest

from fs.errors import DirectoryNotEmpty, FileExpected, ResourceNotFound, \
    ResourceReadOnly
from fs.tempfs import TempFS

import pdart.fs.DeletionSet
from pdart.fs.ReadOnlyFSWithDeletions import ReadOnlyFSWithDeletions


class TestReadOnlyFSWithDeletions(unittest.TestCase):
    def setUp(self):
        self.base_fs = TempFS()
        self.base_fs.settext(u'/foo', u'foo')
        self.base_fs.settext(u'/bar', u'bar')
        self.base_fs.makedirs(u'/dirs/foo')
        self.base_fs.makedirs(u'/dirs/bar')
        self.base_fs.settext(u'/dirs/bar/baz', u'baz')
        self.deletions = pdart.fs.DeletionSet.DeletionSet()
        self.deletions.delete(u'/foo')
        self.deletions.delete(u'/dirs/foo')
        self.fs = ReadOnlyFSWithDeletions(self.base_fs, self.deletions)

    def test_getinfo(self):
        self.assertEqual(self.fs.getinfo(u'/bar'),
                         self.base_fs.getinfo(u'/bar'))

        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.getinfo(u'/quux')

        with self.assertRaises(ResourceNotFound):
            self.fs.getinfo(u'/foo')

        # now check changes
        self.fs.getinfo(u'/bar')
        self.fs.remove(u'/bar')
        with self.assertRaises(ResourceNotFound):
            self.fs.getinfo(u'/bar')

    def test_listdir(self):
        self.assertEqual(set([u'bar', u'dirs']), set(self.fs.listdir(u'/')))

        self.fs.remove(u'/bar')
        self.assertEqual(set([u'dirs']), set(self.fs.listdir(u'/')))

    def test_makedir(self):
        # really doesn't exist
        with self.assertRaises(ResourceNotFound):
            self.fs.makedir(u'/folders/bar')

        # was deleted
        with self.assertRaises(ResourceNotFound):
            self.fs.makedir(u'/dirs/foo/bar')

        # still won't let you
        with self.assertRaises(ResourceReadOnly):
            self.fs.makedir(u'/dirs/quux')

    def test_openbin(self):
        # really doesn't exist
        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/quux')

        # was deleted
        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/foo')

        # exists; writing
        with self.assertRaises(ResourceReadOnly):
            self.fs.openbin(u'/bar', 'w')

        # exists; reading
        with self.fs.openbin(u'/bar', 'r'):
            pass

        # parent dir doesn't exist
        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/quux/foo', 'w')

        # parent dir exists, file doesn't
        with self.assertRaises(ResourceReadOnly):
            self.fs.openbin(u'/dirs/bar/beep', 'w')

        # parent dir exists, file does
        with self.assertRaises(ResourceReadOnly):
            self.fs.openbin(u'/dirs/bar/baz', 'w')

    def test_remove(self):
        # never exists
        with self.assertRaises(ResourceNotFound):
            self.fs.remove(u'/quux')

        # already deleted
        with self.assertRaises(ResourceNotFound):
            self.fs.remove(u'/foo')

        # not a file
        with self.assertRaises(FileExpected):
            self.fs.remove(u'/dirs/bar')

        # exists
        self.fs.remove(u'/bar')
        self.assertFalse(self.fs.exists(u'/bar'))

    def test_removedir(self):
        with self.assertRaises(DirectoryNotEmpty):
            self.fs.removedir(u'/dirs/bar')

        self.fs.remove(u'/dirs/bar/baz')
        self.fs.removedir(u'/dirs/bar')
        self.assertFalse(self.fs.exists(u'/dirs/bar'))

        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.removedir(u'/quux')

        with self.assertRaises(ResourceNotFound):
            self.fs.removedir(u'/dirs/foo')

    def test_setinfo(self):
        # really doesn't exist
        with self.assertRaises(ResourceNotFound):
            self.fs.setinfo(u'/quux', {})

        # was deleted
        with self.assertRaises(ResourceNotFound):
            self.fs.setinfo(u'/foo', {})

        # exists; writing
        with self.assertRaises(ResourceReadOnly):
            self.fs.setinfo(u'/bar', {})

    def test_getsyspath(self):
        for path in [u'/', u'/foo', u'/bar', u'dirs/foo', u'dirs/bar',
                     u'dirs/bar/baz/', u'/i/dont/exist']:
            self.assertEquals(self.base_fs.getsyspath(path),
                              self.fs.getsyspath(path))

import unittest

from fs.tempfs import TempFS

from pdart.fs.DeletionPredicate import *
from pdart.fs.FSWithDeletions import *


class TestDeletionPredicate(DeletionPredicate):
    def is_deleted(self, path):
        # type: (unicode) -> bool
        return 'foo' in path.split('/')


class TestFSWithDeletions(unittest.TestCase):
    def setUp(self):
        self.base_fs = TempFS()
        self.base_fs.settext(u'/foo', u'foo')
        self.base_fs.settext(u'/bar', u'bar')
        self.base_fs.makedirs(u'/dirs/bar')
        self.base_fs.makedirs(u'/dirs/foo')
        self.deletions = TestDeletionPredicate()
        self.fs = FSWithDeletions(self.base_fs, self.deletions)

    # essentials: openbin(),

    def test_getinfo(self):
        self.assertEqual(self.fs.getinfo(u'/bar'),
                         self.base_fs.getinfo(u'/bar'))

        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.getinfo(u'/quux')

        with self.assertRaises(ResourceNotFound):
            self.fs.getinfo(u'/foo')

    def test_listdir(self):
        self.assertEqual(set(self.fs.listdir(u'/')), set([u'bar', u'dirs']))

    def test_makedir(self):
        self.fs.makedir(u'/dirs/quux')
        self.assertTrue(self.fs.exists(u'/dirs/quux'))

        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.makedir(u'/folders/bar')

        with self.assertRaises(ResourceNotFound):
            self.fs.makedir(u'/dirs/foo/bar')

    def test_openbin(self):
        # first, verify the standard result for reading
        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/quux')

        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/foo')

        # first, verify the standard result for writing
        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/quux/quux', 'w')

        # no error raised
        self.fs.openbin(u'/dirs/bar/x', 'w')

        with self.assertRaises(ResourceNotFound):
            self.fs.openbin(u'/dirs/foo/x', 'w')

    def test_remove(self):
        self.fs.remove(u'/bar')
        self.assertFalse(self.fs.exists(u'/bar'))

        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.remove(u'/quux')

        with self.assertRaises(ResourceNotFound):
            self.fs.remove(u'/foo')

    def test_removedir(self):
        self.fs.removedir(u'/dirs/bar')
        self.assertFalse(self.fs.exists(u'/dirs/bar'))

        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.removedir(u'/quux')

        with self.assertRaises(ResourceNotFound):
            self.fs.removedir(u'/dirs/foo')

    def test_setinfo(self):
        # first, verify the standard result
        with self.assertRaises(ResourceNotFound):
            self.fs.setinfo(u'/quux', {})

        with self.assertRaises(ResourceNotFound):
            self.fs.setinfo(u'/foo', {})

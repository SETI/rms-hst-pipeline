import unittest

from fs.memoryfs import MemoryFS
import fs.path
from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.cowfs.COWFS import COWFS


class TestCOWFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        # type: () -> COWFS
        self.tempfs = TempFS()
        return COWFS(self.tempfs)

    def test_makedir_bug(self):
        # These two commands disclosed a bug in COWFS.makedir().  This
        # serves as a regression test.
        self.tempfs.makedir(u'/b$')
        self.fs.makedir(u'/b$/c$')
        self.fs.invariant()

    def test_openbin_bug(self):
        # These two commands disclosed a bug in COWFS.openbin().  This
        # serves as a regression test.
        self.tempfs.makedir(u'/b$')
        self.fs.writetext(u'/b$/c.txt', u'Buggy?')
        self.fs.invariant()


    def test_listdir(self):
        fs = MemoryFS()
        fs.makedirs(u'/b$')
        fs.makedirs(u'/b$/dir1')
        fs.makedirs(u'/b$/dir2')
        fs.writetext(u'/b$/file1.txt', u'file1')
        fs.writetext(u'/b$/file2.txt', u'file2')
        fs.writetext(u'/b$/dir1/file1.txt', u'file1')
        fs.writetext(u'/b$/dir1/file2.txt', u'file2')
        fs.writetext(u'/b$/dir2/file1.txt', u'file1')
        fs.writetext(u'/b$/dir2/file2.txt', u'file2')

        c = COWFS(fs)
        path = u'/b$/dir1/file2.txt'
        c.writetext(path, u'xxxx')

        # Now the COW version is different.  But it should still have
        # the old unchanged files.

        self.assertTrue(c.exists(u'/b$/dir1/file1.txt'))  # Yes, but...
        self.assertEquals({ u'dir1', u'dir2', u'file1.txt', u'file2.txt'},
                          set(c.listdir(u'/b$')))

    def test_getsyspath(self):
        dirpath = u'/b$/dir1'
        self.tempfs.makedirs(dirpath)
        filepath = fs.path.join(dirpath, u'foo.txt')
        self.tempfs.writetext(filepath, u'original contents')

        # syspath for a filepath is the same as the syspath in the
        # basefs.
        self.assertEquals(self.fs.base_fs.getsyspath(filepath),
                          self.fs.getsyspath(filepath))

        # After writing to it, the syspath is now the same as the
        # syspath in the additions_fs.
        self.fs.writetext(filepath, u'replacement contents')
        self.assertEquals(self.fs.additions_fs.getsyspath(filepath),
                          self.fs.getsyspath(filepath))

        # root raises an exception
        with self.assertRaises(fs.errors.NoSysPath):
            self.fs.getsyspath(u'/')

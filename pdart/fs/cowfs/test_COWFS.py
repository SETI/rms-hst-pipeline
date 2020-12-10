import unittest

import fs.path
from fs.memoryfs import MemoryFS
from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.cowfs.COWFS import COWFS


class TestCOWFS(FSTestCases, unittest.TestCase):
    def make_fs(self) -> COWFS:
        self.tempfs = TempFS()
        return COWFS(self.tempfs)

    def test_makedir_bug(self) -> None:
        # These two commands disclosed a bug in COWFS.makedir().  This
        # serves as a regression test.
        self.tempfs.makedir("/b$")
        self.fs.makedir("/b$/c$")
        self.fs.invariant()

    def test_openbin_bug(self) -> None:
        # These two commands disclosed a bug in COWFS.openbin().  This
        # serves as a regression test.
        self.tempfs.makedir("/b$")
        self.fs.writetext("/b$/c.txt", "Buggy?")
        self.fs.invariant()

    @unittest.skip("newly discovered, not worked on yet")
    def test_remove_bug(self) -> None:
        # These two commands disclosed a bug in COWFS.openbin().  This
        # serves as a regression test.
        self.tempfs.makedirs("/b/d")
        self.tempfs.writetext("/b/d/c1.txt", "hmmm")
        self.tempfs.writetext("/b/d/c2.txt", "hmmm")
        self.tempfs.writetext("/b/d/c3.txt", "hmmm")
        self.fs.remove("/b/d/c1.txt")
        self.assertEqual({"c2.txt", "c3.txt"}, set(self.fs.listdir("/b/d")))

    def test_listdir(self) -> None:
        fs = MemoryFS()
        fs.makedirs("/b$")
        fs.makedirs("/b$/dir1")
        fs.makedirs("/b$/dir2")
        fs.writetext("/b$/file1.txt", "file1")
        fs.writetext("/b$/file2.txt", "file2")
        fs.writetext("/b$/dir1/file1.txt", "file1")
        fs.writetext("/b$/dir1/file2.txt", "file2")
        fs.writetext("/b$/dir2/file1.txt", "file1")
        fs.writetext("/b$/dir2/file2.txt", "file2")

        c = COWFS(fs)
        path = "/b$/dir1/file2.txt"
        c.writetext(path, "xxxx")

        # Now the COW version is different.  But it should still have
        # the old unchanged files.

        self.assertTrue(c.exists("/b$/dir1/file1.txt"))  # Yes, but...
        self.assertEqual(
            {"dir1", "dir2", "file1.txt", "file2.txt"}, set(c.listdir("/b$"))
        )

    def test_getsyspath(self) -> None:
        dirpath = "/b$/dir1"
        self.tempfs.makedirs(dirpath)
        filepath = fs.path.join(dirpath, "foo.txt")
        self.tempfs.writetext(filepath, "original contents")

        # syspath for a filepath is the same as the syspath in the
        # basefs.
        self.assertEqual(
            self.fs.base_fs.getsyspath(filepath), self.fs.getsyspath(filepath)
        )

        # After writing to it, the syspath is now the same as the
        # syspath in the additions_fs.
        self.fs.writetext(filepath, "replacement contents")
        self.assertEqual(
            self.fs.additions_fs.getsyspath(filepath), self.fs.getsyspath(filepath)
        )

        # root raises an exception
        with self.assertRaises(fs.errors.NoSysPath):
            self.fs.getsyspath("/")

import shutil
import tempfile
import unittest

import fs.errors
from fs.test import FSTestCases

from pdart.fs.primitives.V1FS import V1FS, V1Primitives
from pdart.fs.primitives.test_FSPrimitives import FSPrimitives_TestBase


class Test_V1Primitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.fs = V1Primitives(self.tmpdir)

    def get_fs(self) -> V1Primitives:
        return self.fs

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class Test_V1FS(FSTestCases, unittest.TestCase):
    def make_fs(self) -> V1FS:
        self.tmpdir = tempfile.mkdtemp()
        return V1FS(self.tmpdir)

    # We rewrite this test to use a shallower hierarchy.
    def test_copydir(self) -> None:
        self.fs.makedirs("foo/bar/egg")
        self.fs.settext("foo/bar/foofoo.txt", "Hello")
        self.fs.makedir("foo2")
        self.fs.copydir("foo/bar", "foo2")
        self.assert_text("foo2/foofoo.txt", "Hello")
        self.assert_isdir("foo2/egg")
        self.assert_text("foo/bar/foofoo.txt", "Hello")
        self.assert_isdir("foo/bar/egg")

        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.copydir("foo", "foofoo")
        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.copydir("spam", "egg", create=True)
        with self.assertRaises(fs.errors.DirectoryExpected):
            self.fs.copydir("foo2/foofoo.txt", "foofoo.txt", create=True)

    # We rewrite this test to use a shallower hierarchy.
    def test_movedir(self) -> None:
        self.fs.makedirs("foo/bar/egg")
        self.fs.settext("foo/bar/foofoo.txt", "Hello")
        self.fs.makedir("foo2")
        self.fs.movedir("foo/bar", "foo2")
        self.assert_text("foo2/foofoo.txt", "Hello")
        self.assert_isdir("foo2/egg")
        self.assert_not_exists("foo/bar/foofoo.txt")
        self.assert_not_exists("foo/bar/egg")

        # Check moving to an unexisting directory
        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.movedir("foo", "foofoo")

        # Check moving an unexisting directory
        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.movedir("spam", "egg", create=True)

        # Check moving a file
        with self.assertRaises(fs.errors.DirectoryExpected):
            self.fs.movedir("foo2/foofoo.txt", "foo2/egg")

    # We rewrite this test to use a shallower hierarchy.
    def test_removetree(self) -> None:
        self.fs.makedirs("foo/bar/baz")
        self.fs.makedirs("foo/egg")
        self.fs.makedirs("foo/a/b")
        self.fs.create("foo/egg.txt")
        self.fs.create("foo/bar/egg.bin")
        self.fs.create("foo/bar/baz/egg.txt")
        self.fs.create("foo/a/b/1.txt")
        self.fs.create("foo/a/b/2.txt")
        self.fs.create("foo/a/b/3.txt")

        self.assert_exists("foo/egg.txt")
        self.assert_exists("foo/bar/egg.bin")

        self.fs.removetree("foo")
        self.assert_not_exists("foo")

    # We rewrite this test to use a shallower hierarchy.
    def test_getsyspath(self) -> None:
        self.fs.create("foo")
        try:
            syspath = self.fs.getsyspath("foo")
        except fs.errors.NoSysPath:
            self.assertFalse(self.fs.hassyspath("foo"))
        else:
            self.assertIsInstance(syspath, str)
            self.assertIsInstance(self.fs.getospath("foo"), bytes)
            self.assertTrue(self.fs.hassyspath("foo"))
        # Should not throw an error
        self.fs.hassyspath("a/b/c/foo")

    # These non-overriding overrides exist only to provide type
    # signatures that are missing in the parent classes.

    def assert_exists(self, path: str) -> None:
        super().assert_exists(path)  # type: ignore

    def assert_isdir(self, path: str) -> None:
        super().assert_isdir(path)  # type: ignore

    def assert_not_exists(self, path: str) -> None:
        super().assert_not_exists(path)  # type: ignore

    def assert_text(self, path: str, txt: str) -> None:
        super().assert_text(path, txt)  # type: ignore

import os.path
import shutil
import tempfile
import unittest

from fs.test import FSTestCases

from pdart.fs.V1FS import *
from pdart.fs.test_FSPrimitives import FSPrimitives_TestBase


class Test_V1Primitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        self.tmpdir = tempfile.mkdtemp()
        self.fs = V1Primitives(self.tmpdir)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        shutil.rmtree(self.tmpdir)


class Test_V1FS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        self.tmpdir = tempfile.mkdtemp()
        return V1FS(self.tmpdir)

    # We rewrite this test to use a shallower hierarchy.
    def test_copydir(self):
        self.fs.makedirs(u'foo/bar/egg')
        self.fs.settext(u'foo/bar/foofoo.txt', u'Hello')
        self.fs.makedir(u'foo2')
        self.fs.copydir(u'foo/bar', u'foo2')
        self.assert_text(u'foo2/foofoo.txt', u'Hello')
        self.assert_isdir(u'foo2/egg')
        self.assert_text(u'foo/bar/foofoo.txt', u'Hello')
        self.assert_isdir(u'foo/bar/egg')

        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.copydir(u'foo', u'foofoo')
        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.copydir(u'spam', u'egg', create=True)
        with self.assertRaises(fs.errors.DirectoryExpected):
            self.fs.copydir(u'foo2/foofoo.txt', u'foofoo.txt', create=True)

    # We rewrite this test to use a shallower hierarchy.
    def test_movedir(self):
        self.fs.makedirs(u'foo/bar/egg')
        self.fs.settext(u'foo/bar/foofoo.txt', u'Hello')
        self.fs.makedir(u'foo2')
        self.fs.movedir(u'foo/bar', u'foo2')
        self.assert_text(u'foo2/foofoo.txt', u'Hello')
        self.assert_isdir(u'foo2/egg')
        self.assert_not_exists(u'foo/bar/foofoo.txt')
        self.assert_not_exists(u'foo/bar/egg')

        # Check moving to an unexisting directory
        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.movedir(u'foo', u'foofoo')

        # Check moving an unexisting directory
        with self.assertRaises(fs.errors.ResourceNotFound):
            self.fs.movedir(u'spam', u'egg', create=True)

        # Check moving a file
        with self.assertRaises(fs.errors.DirectoryExpected):
            self.fs.movedir(u'foo2/foofoo.txt', u'foo2/egg')

import unittest

from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.UsersFS import *
from pdart.fs.test_FSPrimitives import FSPrimitives_TestBase


class Test_UsersPrimitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        self.base_fs = TempFS()
        self.fs = UsersPrimitives(self.base_fs)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        self.base_fs.close()


class Test_UsersFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return UsersFS(TempFS())

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

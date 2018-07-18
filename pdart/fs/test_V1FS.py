import os.path
import shutil
import unittest

from fs.test import FSTestCases

from pdart.fs.V1FS import *
from pdart.fs.test_FSPrimitives import FSPrimitives_TestBase

_TMP_DIR = os.path.abspath('tmp_v1_prims')


class Test_V1Primitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        try:
            os.mkdir(_TMP_DIR)
        except OSError:
            shutil.rmtree(_TMP_DIR)
            os.mkdir(_TMP_DIR)

        self.fs = V1Primitives(_TMP_DIR)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        shutil.rmtree(_TMP_DIR)


class Test_V1FS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        try:
            os.mkdir(_TMP_DIR)
        except OSError:
            shutil.rmtree(_TMP_DIR)
            os.mkdir(_TMP_DIR)
        return V1FS(_TMP_DIR)

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

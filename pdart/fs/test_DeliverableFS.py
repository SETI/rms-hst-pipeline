import unittest

from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.DeliverableFS import *
from pdart.fs.test_FSPrimitives import FSPrimitives_TestBase


# @unittest.skip('')
class Test_DeliverablePrimitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self):
        # type: () -> None
        self.base_fs = TempFS()
        self.fs = DeliverablePrimitives(self.base_fs)

    def get_fs(self):
        return self.fs

    def tearDown(self):
        # type: () -> None
        self.base_fs.close()

    # We expand these tests to hit the boundary cases: around product
    # directories.

    def test_add_child_dir(self):
        # type: () -> None
        FSPrimitives_TestBase.test_add_child_dir(self)
        fs = self.fs
        root = fs.root_node()

        # create PDART archive dirs
        bundle_dir = self._assert_add_child_dir_is_correct(root,
                                                           'hst_06141')
        collection_dir = self._assert_add_child_dir_is_correct(
            bundle_dir, 'data_wfpc2_cmh')
        product_dir = self._assert_add_child_dir_is_correct(collection_dir,
                                                            'u2mu0101j')

        # create a random dir
        b_dir = self._assert_add_child_dir_is_correct(root, 'b')
        c_dir = self._assert_add_child_dir_is_correct(b_dir, 'c')
        p_dir = self._assert_add_child_dir_is_correct(c_dir, 'p')
        d_dir = self._assert_add_child_dir_is_correct(p_dir, 'd')

    def test_add_child_file(self):
        # type: () -> None
        FSPrimitives_TestBase.test_add_child_dir(self)
        fs = self.fs
        root = fs.root_node()

        # create PDART archive files
        bundle_dir = fs.add_child_dir(root, 'hst_06141')
        bundle_label = self._assert_add_child_file_is_correct(bundle_dir,
                                                              'bundle.xml')

        collection_dir = fs.add_child_dir(bundle_dir, 'data_wfpc2_cmh')
        collection_label = self._assert_add_child_file_is_correct(
            collection_dir,
            'collection.xml')

        product_dir = fs.add_child_dir(collection_dir, 'u2mu0101j')
        product_file = self._assert_add_child_file_is_correct(
            product_dir, 'u2mu0101j_cmh.fits')
        product_label = self._assert_add_child_file_is_correct(
            product_dir, 'u2mu0101j_cmh.xml')

        # create a random file
        b_dir = fs.add_child_dir(root, 'b')
        b_file = self._assert_add_child_file_is_correct(b_dir, 'b.txt')
        c_dir = fs.add_child_dir(b_dir, 'c')
        c_file = self._assert_add_child_file_is_correct(c_dir, 'c.txt')
        p_dir = fs.add_child_dir(c_dir, 'p')
        p_file = self._assert_add_child_file_is_correct(p_dir, 'p.txt')
        d_dir = fs.add_child_dir(p_dir, 'd')
        d_file = self._assert_add_child_file_is_correct(d_dir, 'd.txt')

    def test_remove_child(self):
        # type: () -> None
        FSPrimitives_TestBase.test_remove_child(self)
        fs = self.fs

        # create PDART archive files
        root = fs.root_node()
        bundle_dir = fs.add_child_dir(root, 'hst_06141')
        bundle_label = fs.add_child_file(bundle_dir,
                                         'bundle.xml')

        collection_dir = fs.add_child_dir(bundle_dir, 'data_wfpc2_cmh')
        collection_label = fs.add_child_file(collection_dir,
                                             'collection.xml')

        product_dir = fs.add_child_dir(collection_dir, 'u2mu0101j')
        product_file = fs.add_child_file(product_dir,
                                         'u2mu0101j_cmh.fits')
        product_label = fs.add_child_file(product_dir,
                                          'u2mu0101j_cmh.xml')

        # delete it, bottom up
        self._assert_remove_child_is_correct(product_dir, 'u2mu0101j_cmh.fits')
        self._assert_remove_child_is_correct(product_dir, 'u2mu0101j_cmh.xml')
        self.assertFalse(fs.get_children(product_dir))
        self._assert_remove_child_is_correct(collection_dir, 'u2mu0101j')
        self._assert_remove_child_is_correct(collection_dir, 'collection.xml')
        self.assertFalse(fs.get_children(collection_dir))
        self._assert_remove_child_is_correct(bundle_dir, 'data_wfpc2_cmh')
        self._assert_remove_child_is_correct(bundle_dir, 'bundle.xml')
        self.assertFalse(fs.get_children(bundle_dir))
        self._assert_remove_child_is_correct(root, 'hst_06141')

        # create random files
        b_dir = fs.add_child_dir(root, 'b')
        b_label = fs.add_child_file(b_dir, 'b.txt')
        c_dir = fs.add_child_dir(b_dir, 'c')
        c_label = fs.add_child_file(c_dir, 'c.txt')
        p_dir = fs.add_child_dir(c_dir, 'p')
        p_file = fs.add_child_file(p_dir, 'p.txt')
        d_dir = fs.add_child_dir(p_dir, 'd')
        d_file = fs.add_child_file(d_dir, 'd.txt')

        # delete it, bottom up
        self._assert_remove_child_is_correct(d_dir, 'd.txt')
        self.assertFalse(fs.get_children(d_dir))
        self._assert_remove_child_is_correct(p_dir, 'd')
        self._assert_remove_child_is_correct(p_dir, 'p.txt')
        self.assertFalse(fs.get_children(p_dir))
        self._assert_remove_child_is_correct(c_dir, 'p')
        self._assert_remove_child_is_correct(c_dir, 'c.txt')
        self.assertFalse(fs.get_children(c_dir))
        self._assert_remove_child_is_correct(b_dir, 'c')
        self._assert_remove_child_is_correct(b_dir, 'b.txt')
        self.assertFalse(fs.get_children(b_dir))
        self._assert_remove_child_is_correct(root, 'b')


@unittest.skip('')
class Test_DeliverableFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return DeliverableFS(TempFS())

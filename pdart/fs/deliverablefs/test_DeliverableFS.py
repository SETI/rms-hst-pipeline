import unittest

from fs.tempfs import TempFS
from fs.test import FSTestCases

from pdart.fs.deliverablefs.DeliverableFS import (
    DeliverableFS,
    DeliverablePrimitives,
    _translate_path_to_base_path,
)
from pdart.fs.primitives.test_FSPrimitives import FSPrimitives_TestBase


class Test_DeliverablePrimitives(unittest.TestCase, FSPrimitives_TestBase):
    def setUp(self) -> None:
        self.base_fs = TempFS()
        self.fs = DeliverablePrimitives(self.base_fs)

    def get_fs(self) -> DeliverablePrimitives:
        return self.fs

    def tearDown(self) -> None:
        self.base_fs.close()

    # We expand these tests to hit the boundary cases: around product
    # directories.

    def test_add_child_dir(self) -> None:
        FSPrimitives_TestBase.test_add_child_dir(self)
        fs = self.fs
        root = fs.root_node()

        # create PDART archive dirs
        bundle_dir = self._assert_add_child_dir_is_correct(root, "hst_06141")
        collection_dir = self._assert_add_child_dir_is_correct(
            bundle_dir, "data_wfpc2_cmh"
        )
        product_dir = self._assert_add_child_dir_is_correct(collection_dir, "u2mu0101j")

        doc_collection_dir = self._assert_add_child_dir_is_correct(
            bundle_dir, "document"
        )
        doc_product_dir = self._assert_add_child_dir_is_correct(
            doc_collection_dir, "phase2"
        )

        # create a random dir
        b_dir = self._assert_add_child_dir_is_correct(root, "b")
        c_dir = self._assert_add_child_dir_is_correct(b_dir, "c")
        p_dir = self._assert_add_child_dir_is_correct(c_dir, "p")
        d_dir = self._assert_add_child_dir_is_correct(p_dir, "d")

    def test_add_child_file(self) -> None:
        FSPrimitives_TestBase.test_add_child_dir(self)
        fs = self.fs
        root = fs.root_node()

        # create PDART archive files
        bundle_dir = fs.add_child_dir(root, "hst_06141")
        bundle_label = self._assert_add_child_file_is_correct(bundle_dir, "bundle.xml")

        collection_dir = fs.add_child_dir(bundle_dir, "data_wfpc2_cmh")
        collection_label = self._assert_add_child_file_is_correct(
            collection_dir, "collection.xml"
        )

        doc_collection_dir = fs.add_child_dir(bundle_dir, "document")
        doc_collection_label = self._assert_add_child_file_is_correct(
            doc_collection_dir, "collection.xml"
        )

        product_dir = fs.add_child_dir(collection_dir, "u2mu0101j")
        product_file = self._assert_add_child_file_is_correct(
            product_dir, "u2mu0101j_cmh.fits"
        )
        product_label = self._assert_add_child_file_is_correct(
            product_dir, "u2mu0101j_cmh.xml"
        )

        doc_product_dir = fs.add_child_dir(doc_collection_dir, "phase2")
        doc_product_file = self._assert_add_child_file_is_correct(
            doc_product_dir, "phase2.pdf"
        )

        # create a random file
        b_dir = fs.add_child_dir(root, "b")
        b_file = self._assert_add_child_file_is_correct(b_dir, "b.txt")
        c_dir = fs.add_child_dir(b_dir, "c")
        c_file = self._assert_add_child_file_is_correct(c_dir, "c.txt")
        p_dir = fs.add_child_dir(c_dir, "p")
        p_file = self._assert_add_child_file_is_correct(p_dir, "p.txt")
        d_dir = fs.add_child_dir(p_dir, "d")
        d_file = self._assert_add_child_file_is_correct(d_dir, "d.txt")

    def test_remove_child(self) -> None:
        FSPrimitives_TestBase.test_remove_child(self)
        fs = self.fs

        # create PDART archive files
        root = fs.root_node()
        bundle_dir = fs.add_child_dir(root, "hst_06141")
        bundle_label = fs.add_child_file(bundle_dir, "bundle.xml")

        collection_dir = fs.add_child_dir(bundle_dir, "data_wfpc2_cmh")
        collection_label = fs.add_child_file(collection_dir, "collection.xml")

        product_dir = fs.add_child_dir(collection_dir, "u2mu0101j")
        product_file = fs.add_child_file(product_dir, "u2mu0101j_cmh.fits")
        product_label = fs.add_child_file(product_dir, "u2mu0101j_cmh.xml")

        # delete it, bottom up
        self._assert_remove_child_is_correct(product_dir, "u2mu0101j_cmh.fits")
        self._assert_remove_child_is_correct(product_dir, "u2mu0101j_cmh.xml")
        self.assertFalse(fs.get_children(product_dir))
        self._assert_remove_child_is_correct(collection_dir, "u2mu0101j")
        self._assert_remove_child_is_correct(collection_dir, "collection.xml")
        self.assertFalse(fs.get_children(collection_dir))
        self._assert_remove_child_is_correct(bundle_dir, "data_wfpc2_cmh")
        self._assert_remove_child_is_correct(bundle_dir, "bundle.xml")
        self.assertFalse(fs.get_children(bundle_dir))
        self._assert_remove_child_is_correct(root, "hst_06141")

        # create random files
        b_dir = fs.add_child_dir(root, "b")
        b_label = fs.add_child_file(b_dir, "b.txt")
        c_dir = fs.add_child_dir(b_dir, "c")
        c_label = fs.add_child_file(c_dir, "c.txt")
        p_dir = fs.add_child_dir(c_dir, "p")
        p_file = fs.add_child_file(p_dir, "p.txt")
        d_dir = fs.add_child_dir(p_dir, "d")
        d_file = fs.add_child_file(d_dir, "d.txt")

        # delete it, bottom up
        self._assert_remove_child_is_correct(d_dir, "d.txt")
        self.assertFalse(fs.get_children(d_dir))
        self._assert_remove_child_is_correct(p_dir, "d")
        self._assert_remove_child_is_correct(p_dir, "p.txt")
        self.assertFalse(fs.get_children(p_dir))
        self._assert_remove_child_is_correct(c_dir, "p")
        self._assert_remove_child_is_correct(c_dir, "c.txt")
        self.assertFalse(fs.get_children(c_dir))
        self._assert_remove_child_is_correct(b_dir, "c")
        self._assert_remove_child_is_correct(b_dir, "b.txt")
        self.assertFalse(fs.get_children(b_dir))
        self._assert_remove_child_is_correct(root, "b")

    def test_translate_path_to_base_path(self) -> None:
        test_cases = [
            ("/b", "/b"),
            ("/b/label.xml", "/b/label.xml"),
            ("/b/c", "/b/c"),
            ("/b/c/label.xml", "/b/c/label.xml"),
            ("/b/document", "/b/document"),
            ("/b/document/label.xml", "/b/document/label.xml"),
            ("/b/c/visit_01", "/b/c/u2mu0101j"),
            ("/b/c/visit_01/label.xml", "/b/c/u2mu0101j/label.xml"),
            ("/b/document/phase2", "/b/document/phase2"),
            ("/b/document/phase2/info.pdf", "/b/document/phase2/info.pdf"),
            ("/b/c/no_visit/p", "/b/c/p"),
            ("/b/c/no_visit/p/label.xml", "/b/c/p/label.xml"),
        ]
        for (expected, pds4_path) in test_cases:
            actual = _translate_path_to_base_path(str(pds4_path))
            self.assertEqual(str(expected), actual)


class Test_DeliverableFS(FSTestCases, unittest.TestCase):
    def make_fs(self) -> DeliverableFS:
        return DeliverableFS(TempFS())

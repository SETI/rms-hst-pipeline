import os
import shutil
import tempfile
import unittest

from pdart.documents.Downloads import download_product_documents


class TestDownloads(unittest.TestCase):
    def setUp(self) -> None:
        self.download_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.download_dir)

    def test_download_product_documents(self) -> None:
        res = download_product_documents(1, self.download_dir)
        expected = {"phase2.pro", "phase2.prop"}
        self.assertEqual(expected, res)
        self.assertEqual(expected, {str(f) for f in os.listdir(self.download_dir)})

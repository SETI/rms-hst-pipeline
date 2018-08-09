import os
import shutil
import tempfile
import unittest

from pdart.documents.Downloads import *


class TestDownloads(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.download_dir = tempfile.mkdtemp()

    def tearDown(self):
        # type: () -> None
        shutil.rmtree(self.download_dir)

    def test_download_product_documents(self):
        # type: () -> None
        res = download_product_documents(1, self.download_dir)
        expected = {u'phase2.pro', u'phase2.prop'}
        self.assertEqual(expected, res)
        self.assertEqual(expected,
                         {unicode(f) for f in os.listdir(self.download_dir)})

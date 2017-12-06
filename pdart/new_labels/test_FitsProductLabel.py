import unittest

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.FitsProductLabel import *


class Test_FitsProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()

    def tearDown(self):
        # type: () -> None
        pass

    @unittest.skip('under construction')
    def test_make_fits_product_label(self):
        # type: () -> None
        label = make_fits_product_label(self.db, 'urn:nasa:pds:b:c:p::2', True)
        self.assertTrue(label)

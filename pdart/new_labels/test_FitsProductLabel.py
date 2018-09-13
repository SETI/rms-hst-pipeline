import unittest

from fs.path import basename

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.FitsProductLabel import make_fits_product_label
from pdart.new_labels.Utils import golden_file_contents, path_to_testfile


class Test_FitsProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()
        self.db.create_tables()

    def test_make_fits_product_label(self):
        # type: () -> None

        bundle_lidvid = \
            'urn:nasa:pds:hst_13012::123.90201'
        self.db.create_bundle(bundle_lidvid)

        collection_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw::3.14159'
        self.db.create_non_document_collection(collection_lidvid,
                                               bundle_lidvid)

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.13'
        self.db.create_fits_product(fits_product_lidvid,
                                    collection_lidvid)

        os_filepath = path_to_testfile('jbz504eoq_raw.fits')

        populate_database_from_fits_file(self.db,
                                         os_filepath,
                                         fits_product_lidvid)

        file_basename = basename(os_filepath)

        label = make_fits_product_label(self.db,
                                        fits_product_lidvid,
                                        file_basename,
                                        True)

        expected = golden_file_contents('test_FitsProductLabel.golden.xml')
        self.assertEqual(expected, label)

import unittest

from fs.path import basename, join

from pdart.new_db.BrowseFileDB import populate_database_from_browse_file
from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.BrowseProductLabel import *
from pdart.new_labels.Utils import golden_file_contents


class Test_BrowseProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()

    def tearDown(self):
        # type: () -> None
        pass

    def test_make_browse_product_label(self):
        # type: () -> None
        self.db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        bundle_lidvid = \
            'urn:nasa:pds:hst_13012::123'
        self.db.create_bundle(bundle_lidvid)

        fits_collection_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw::3.14159'
        self.db.create_non_document_collection(fits_collection_lidvid,
                                               bundle_lidvid)

        browse_collection_lidvid = \
            'urn:nasa:pds:hst_13012:browse_acs_raw::3.14159'
        self.db.create_non_document_collection(browse_collection_lidvid,
                                               bundle_lidvid)

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2.1'
        self.db.create_fits_product(fits_product_lidvid,
                                    fits_collection_lidvid)

        browse_product_lidvid = \
            'urn:nasa:pds:hst_13012:browse_acs_raw:jbz504eoq_raw::2.1'
        self.db.create_browse_product(browse_product_lidvid,
                                      fits_product_lidvid,
                                      browse_collection_lidvid)

        os_filepath = join(
            archive,
            'hst_13012/browse_acs_raw/visit_04/jbz504eoq_raw.jpg')
        browse_file_basename = basename(os_filepath)

        populate_database_from_browse_file(self.db,
                                           browse_product_lidvid,
                                           fits_product_lidvid,
                                           browse_collection_lidvid,
                                           browse_file_basename,
                                           5492356)

        browse_file = self.db.get_file(browse_product_lidvid,
                                       browse_file_basename)

        str = make_browse_product_label(self.db,
                                        browse_product_lidvid,
                                        browse_file_basename,
                                        True)

        expected = golden_file_contents('test_BrowseProductLabel.golden.xml')
        self.assertEqual(expected, str)

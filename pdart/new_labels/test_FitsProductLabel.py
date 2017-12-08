import unittest

from fs.path import basename, join

from pdart.new_db.BundleDB import create_bundle_db_in_memory
from pdart.new_labels.FitsProductLabel import *
from pdart.new_db.FitsFileDB import card_dictionaries, populate_from_fits_file
from pdart.new_db.SqlAlchTables import File, FitsFile
from pdart.xml.Pretty import pretty_print


class Test_FitsProductLabel(unittest.TestCase):
    def setUp(self):
        # type: () -> None
        self.db = create_bundle_db_in_memory()

    def tearDown(self):
        # type: () -> None
        pass

    @unittest.skip("under development")
    def test_make_fits_product_label(self):
        # type: () -> None
        db = create_bundle_db_in_memory()
        db.create_tables()
        archive = '/Users/spaceman/Desktop/Archive'

        fits_product_lidvid = \
            'urn:nasa:pds:hst_13012:data_acs_raw:jbz504eoq_raw::2'
        os_filepath = join(
            archive,
            'hst_13012/data_acs_raw/visit_04/jbz504eoq_raw.fits')

        populate_from_fits_file(db,
                                os_filepath,
                                fits_product_lidvid)

        file_basename = basename(os_filepath)

        fits_file = db.session.query(FitsFile).filter(
            File.product_lidvid == fits_product_lidvid).filter(
            File.basename == file_basename).one()

        hdu_count = fits_file.hdu_count

        card_dicts = card_dictionaries(db.session,
                                       fits_product_lidvid,
                                       hdu_count)

        str = make_fits_product_label(self.db,
                                      card_dicts,
                                      fits_product_lidvid,
                                      True)
        str = pretty_print(str)
        print str

        self.assertTrue(label)

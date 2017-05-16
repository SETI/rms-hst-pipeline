import os.path
import shutil
import tempfile
import unittest

from pdart.db.SqlAlchTables import *
from pdart.pds4labels.SqlAlchLabels import *
from pdart.xml.Schema import verify_label_or_raise


class TestSqlAlch(unittest.TestCase):
    BUNDLE_LID = 'urn:nasa:pds:hst_00666'

    DATA_COLLECTION_LID = 'urn:nasa:pds:hst_00666:data_wfpc2_xxx'
    DATA_COLLECTION_LABEL_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/collection_data.xml"
    DATA_COLLECTION_INVENTORY_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/collection_data.csv"

    BROWSE_COLLECTION_LID = 'urn:nasa:pds:hst_00666:browse_wfpc2_xxx'
    BROWSE_COLLECTION_LABEL_FILEPATH = \
        "/I/don't/exist/hst_00666/browse_wfpc2_xxx/collection_browse.xml"
    BROWSE_COLLECTION_INVENTORY_FILEPATH = \
        "/I/don't/exist/hst_00666/browse_wfpc2_xxx/collection_browse.csv"

    DATA_PRODUCT_LID = 'urn:nasa:pds:hst_00666:data_wfpc2_xxx:u2novv01j_xxx'
    DATA_PRODUCT_FITS_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/visit_vv/u2novv01j_xxx.fits"
    DATA_PRODUCT_LABEL_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/visit_vv/u2novv01j_xxx.xml"

    BROWSE_PRODUCT_LID = \
        'urn:nasa:pds:hst_00666:browse_wfpc2_xxx:u2novv01j_xxx'
    BROWSE_PRODUCT_LABEL_FILEPATH = \
        "/I/don't/exist/hst_00666/browse_wfpc2_xxx/visit_vv/u2novv01j_xxx.xml"
    BROWSE_PRODUCT_JPG_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/visit_vv/u2novv01j_xxx.jpg"

    def setUp(self):
        # type: () -> None
        self.test_dir = tempfile.mkdtemp()

        db_filepath = os.path.join(self.test_dir, 'tmp.db')
        self.session = create_database_tables_and_session(db_filepath)

        self.db_bundle = Bundle(
            lid=self.BUNDLE_LID,
            proposal_id=666,
            archive_path="/I/don't/exist",
            full_filepath="/I/don't/exist/hst_00666",
            label_filepath="/I/don't/exist/hst_00666/bundle.xml")
        self.session.add(self.db_bundle)

        self.db_collection = NonDocumentCollection(
            lid=self.DATA_COLLECTION_LID,
            bundle_lid=self.BUNDLE_LID,
            prefix='data',
            suffix='xxx',
            instrument='wfpc2',
            full_filepath="/I/don't/exist/hst_00666/data_wfpc2_xxx",
            label_filepath=self.DATA_COLLECTION_LABEL_FILEPATH,
            inventory_name='collection_data.csv',
            inventory_filepath=self.DATA_COLLECTION_INVENTORY_FILEPATH
            )
        self.session.add(self.db_collection)

        self.db_fits_product = FitsProduct(
            lid=self.DATA_PRODUCT_LID,
            collection_lid=self.DATA_COLLECTION_LID,
            fits_filepath=self.DATA_PRODUCT_FITS_FILEPATH,
            label_filepath=self.DATA_PRODUCT_LABEL_FILEPATH,
            visit='vv'
            )
        self.session.add(self.db_fits_product)

        db_hdu = Hdu(product_lid=self.DATA_PRODUCT_LID,
                     hdu_index=0,
                     hdr_loc=0,
                     dat_loc=1024,
                     dat_span=666)
        self.session.add(db_hdu)

        cards = [('TARGNAME', 'JUPITER'),
                 ('EXPTIME', '3.14159265'),
                 ('DATE-OBS', '2012-12-25'),
                 ('TIME-OBS', '17:18:19.2021'),
                 ('NAXIS', '2'),
                 ('NAXIS1', '6'),
                 ('NAXIS2', '111'),
                 ('BITPIX', '8')]
        for (k, v) in cards:
            db_card = Card(product_lid=self.DATA_PRODUCT_LID,
                           hdu_index=0,
                           keyword=k,
                           value=v)
            self.session.add(db_card)

        self.db_browse_collection = NonDocumentCollection(
            lid=self.BROWSE_COLLECTION_LID,
            bundle_lid=self.BUNDLE_LID,
            prefix='browse',
            suffix='xxx',
            instrument='wfpc2',
            full_filepath="/I/don't/exist/hst_00666/browse_wfpc2_xxx",
            label_filepath=self.BROWSE_COLLECTION_LABEL_FILEPATH,
            inventory_name='collection_browse.csv',
            inventory_filepath=self.BROWSE_COLLECTION_INVENTORY_FILEPATH
            )
        self.session.add(self.db_browse_collection)

        self.db_browse_product = BrowseProduct(
            lid=self.BROWSE_PRODUCT_LID,
            collection_lid=self.BROWSE_COLLECTION_LID,
            label_filepath=self.BROWSE_PRODUCT_LABEL_FILEPATH,
            browse_filepath=self.BROWSE_PRODUCT_JPG_FILEPATH,
            object_length=12345
            )
        self.session.add(self.db_browse_product)

        self.session.commit()

    def tearDown(self):
        # type: () -> None
        self.session.close()
        shutil.rmtree(self.test_dir)

    def test_make_product_bundle_label(self):
        # type: () -> None

        # It requires an argument
        with self.assertRaises(AssertionError):
            make_product_bundle_label(None)

        label = make_product_bundle_label(self.db_bundle)
        verify_label_or_raise(label)

    def test_make_product_collection_label(self):
        # type: () -> None

        # It requires an argument
        with self.assertRaises(AttributeError):
            make_product_collection_label(None)

        label = make_product_collection_label(self.db_collection)
        verify_label_or_raise(label)

    def test_make_product_observational_label(self):
        # type: () -> None

        # It requires an argument
        with self.assertRaises(AssertionError):
            make_product_observational_label(None)

        label = make_product_observational_label(self.db_fits_product)
        verify_label_or_raise(label)

    def test_make_product_browse_label(self):
        # type: () -> None

        # It requires two arguments
        with self.assertRaises(AttributeError):
            make_product_browse_label(None, self.db_browse_product)
        with self.assertRaises(AssertionError):
            make_product_browse_label(self.db_browse_collection, None)
        with self.assertRaises(AssertionError):
            make_product_browse_label(None, None)

        label = make_product_browse_label(self.db_browse_collection,
                                          self.db_browse_product)
        verify_label_or_raise(label)

if __name__ == '__main__':
    unittest.main()

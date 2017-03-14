import os.path
import shutil
import tempfile
import unittest

from pdart.xml.Schema import verify_label_or_raise

from SqlAlchLabels import *
from SqlAlchTables import *


class TestSqlAlch(unittest.TestCase):
    BUNDLE_LID = 'urn:nasa:pds:hst_00666'
    COLLECTION_LID = 'urn:nasa:pds:hst_00666:data_wfpc2_xxx'
    COLLECTION_LABEL_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/collection_data.xml"
    COLLECTION_INVENTORY_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/collection_data.csv"
    PRODUCT_LID = 'urn:nasa:pds:hst_00666:data_wfpc2_xxx:u2novv01j_xxx'
    PRODUCT_FITS_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/visit_vv/u2novv01j_xxx.fits"
    PRODUCT_LABEL_FILEPATH = \
        "/I/don't/exist/hst_00666/data_wfpc2_xxx/visit_vv/u2novv01j_xxx.xml"

    def setUp(self):
        # type: () -> None
        self.test_dir = tempfile.mkdtemp()

        db_filepath = os.path.join(self.test_dir, 'tmp.db')
        engine = create_engine('sqlite:///' + db_filepath)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        self.db_bundle = Bundle(
            lid=self.BUNDLE_LID,
            proposal_id=666,
            archive_path="/I/don't/exist",
            full_filepath="/I/don't/exist/hst_00666",
            label_filepath="/I/don't/exist/hst_00666/bundle.xml")
        self.session.add(self.db_bundle)

        self.db_collection = Collection(
            lid=self.COLLECTION_LID,
            bundle_lid=self.BUNDLE_LID,
            prefix='data',
            suffix='xxx',
            instrument='wfpc2',
            full_filepath="/I/don't/exist/hst_00666/data_wfpc2_xxx",
            label_filepath=self.COLLECTION_LABEL_FILEPATH,
            inventory_name='collection_data.csv',
            inventory_filepath=self.COLLECTION_INVENTORY_FILEPATH
            )
        self.session.add(self.db_collection)

        self.db_product = FitsProduct(
            lid=self.PRODUCT_LID,
            collection_lid=self.COLLECTION_LID,
            fits_filepath=self.PRODUCT_FITS_FILEPATH,
            label_filepath=self.PRODUCT_LABEL_FILEPATH,
            visit='vv'
            )
        self.session.add(self.db_product)

        db_hdu = Hdu(product_lid=self.PRODUCT_LID,
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
            db_card = Card(product_lid=self.PRODUCT_LID,
                           hdu_index=0,
                           keyword=k,
                           value=v)
            self.session.add(db_card)

        self.session.commit()

    def tearDown(self):
        # type: () -> None
        self.session.close()
        shutil.rmtree(self.test_dir)

    def test_make_product_bundle_label(self):
        # type: () -> None
        label = make_product_bundle_label(self.db_bundle)
        verify_label_or_raise(label)

    def test_make_product_collection_label(self):
        # type: () -> None
        label = make_product_collection_label(self.db_collection)
        verify_label_or_raise(label)

    def test_make_product_observational_label(self):
        # type: () -> None
        label = make_product_observational_label(self.db_product)
        print label
        verify_label_or_raise(label)

if __name__ == '__main__':
    unittest.main()

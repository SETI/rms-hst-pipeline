from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker

from pdart.new_db.SqlAlchTables import *
from pdart.pds4.LIDVID import LIDVID

_BUNDLE_DB_NAME = 'bundle$database.db'  # type: unicode


class BundleDB(object):
    def __init__(self, url):
        # type: (unicode) -> None
        self.url = url
        self.engine = create_engine(url)
        self.session = sessionmaker(bind=self.engine)()

    @staticmethod
    def create_database_from_os_filepath(os_filepath):
        # type: (unicode) -> BundleDB
        return BundleDB('sqlite:///' + os_filepath)

    @staticmethod
    def create_database_in_memory():
        # type: () -> BundleDB
        return BundleDB('sqlite:///')

    def create_tables(self):
        # type: () -> None
        create_tables(self.engine)

    def bundle_exists(self, bundle_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a bundle with the given LIDVID exists in the database.
        """
        return self.session.query(
            exists().where(Bundle.lidvid == bundle_lidvid)).scalar()

    def collection_exists(self, collection_lidvid):
        # type: (str) -> bool

        """
        Returns True iff a collection with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(Collection.lidvid == collection_lidvid)).scalar()

    def non_document_collection_exists(self, collection_lidvid):
        # type: (str) -> bool

        """
        Returns True iff a non-document collection with the given
        LIDVID exists in the database.
        """
        return self.session.query(
            exists().where(
                NonDocumentCollection.collection_lidvid ==
                collection_lidvid)).scalar()

    def product_exists(self, product_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a product with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(Product.lidvid == product_lidvid)).scalar()

    def fits_product_exists(self, product_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a FITS product with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(
                FitsProduct.product_lidvid == product_lidvid)).scalar()

    def file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a file with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(File.basename == basename).where(
                File.product_lidvid == product_lidvid)).scalar()

    def fits_file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a FITS file with the given LIDVID exists in
        the database.
        """
        return self.session.query(
            exists().where(FitsFile.basename == basename).where(
                FitsFile.product_lidvid == product_lidvid).where(
                File.type == 'fits_file')).scalar()

    def bad_fits_file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a bad FITS file record with the given LIDVID
        exists in the database.
        """
        return self.session.query(
            exists().where(BadFitsFile.basename == basename).where(
                BadFitsFile.product_lidvid == product_lidvid).where(
                File.type == 'bad_fits_file')).scalar()

    def hdu_exists(self, index, basename, product_lidvid):
        # type: (int, unicode, str) -> bool
        """
        Returns True iff the n-th HDU for that FITS file exists
        """
        return self.session.query(
            exists().where(
                Product.lidvid == product_lidvid).where(
                Hdu.hdu_index == index)).scalar()

    def create_bundle(self, bundle_lidvid):
        # type: (str) -> None
        """
        Create a bundle with this LIDVID if none exists.
        """
        LIDVID(bundle_lidvid)
        if not self.bundle_exists(bundle_lidvid):
            self.session.add(Bundle(lidvid=bundle_lidvid))

    def create_non_document_collection(self, collection_lidvid, bundle_lidvid):
        # type: (str, str) -> None
        """
        Create a non_document_collection with this LIDVID if none exists.
        """
        LIDVID(collection_lidvid)
        LIDVID(bundle_lidvid)
        if self.collection_exists(collection_lidvid):
            if self.non_document_collection_exists(collection_lidvid):
                pass
            else:
                raise Exception(
                    'non-non-document-collection with LIDVID %s already exists'
                    % collection_lidvid)
        else:
            self.session.add(
                NonDocumentCollection(
                    lidvid=collection_lidvid,
                    collection_lidvid=collection_lidvid,
                    bundle_lidvid=bundle_lidvid))

    def create_fits_product(self, product_lidvid, collection_lidvid):
        # type: (str, str) -> None
        """
        Create a product with this LIDVID if none exists.
        """
        LIDVID(product_lidvid)
        LIDVID(collection_lidvid)
        if self.product_exists(product_lidvid):
            if self.fits_product_exists(product_lidvid):
                pass
            else:
                raise Exception(
                    'non-FITS product with LIDVID %s already exists' %
                    product_lidvid)
        else:
            self.session.add(
                FitsProduct(lidvid=product_lidvid,
                            collection_lidvid=collection_lidvid))

    def create_fits_file(self, basename, product_lidvid):
        # type: (unicode, str) -> None
        """
        Create a FITS file with this basename belonging to the product
        if none exists.
        """
        LIDVID(product_lidvid)
        if self.fits_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                FitsFile(basename=basename,
                         product_lidvid=product_lidvid))

    def create_bad_fits_file(self, basename, product_lidvid,
                             exception_message):
        # type: (unicode, str, str) -> None
        """
        Create a bad FITS file record with this basename belonging to
        the product if none exists.
        """
        LIDVID(product_lidvid)
        if self.fits_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                BadFitsFile(basename=basename,
                            product_lidvid=product_lidvid,
                            exception_message=exception_message))

    def close(self):
        # type: () -> None
        """
        Close the session associated with this BundleDB.
        """
        self.session.close()
        self.session = None

    def is_open(self):
        # type: () -> bool
        """
        Return True iff the session associated with this BundleDB has not
        been closed.
        """
        return self.session is not None

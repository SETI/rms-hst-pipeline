import re

from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker

import pdart.pds4.Bundle
import pdart.pds4.Collection
from pdart.new_db.SqlAlchTables import *
from pdart.pds4.LIDVID import LIDVID

_BUNDLE_DB_NAME = 'bundle$database.db'  # type: unicode


def create_bundle_db_from_os_filepath(os_filepath):
    # type: (unicode) -> BundleDB
    return BundleDB('sqlite:///' + os_filepath)


def create_bundle_db_in_memory():
    # type: () -> BundleDB
    return BundleDB('sqlite:///')


def _lidvid_to_proposal_id(bundle_lidvid):
    lid = LIDVID(bundle_lidvid).lid()
    bundle_id = lid.bundle_id
    return int(re.match(pdart.pds4.Bundle.Bundle.DIRECTORY_PATTERN,
                        bundle_id).group(1))


def _lidvid_to_instrument(nondoc_collection_lidvid):
    lid = LIDVID(nondoc_collection_lidvid).lid()
    collection_id = lid.collection_id
    return re.match(pdart.pds4.Collection.Collection.DIRECTORY_PATTERN,
                    collection_id).group(3)


def _lidvid_to_suffix(nondoc_collection_lidvid):
    lid = LIDVID(nondoc_collection_lidvid).lid()
    collection_id = lid.collection_id
    return re.match(pdart.pds4.Collection.Collection.DIRECTORY_PATTERN,
                    collection_id).group(4)


class BundleDB(object):
    def __init__(self, url):
        # type: (unicode) -> None
        self.url = url
        self.engine = create_engine(url)
        self.session = sessionmaker(bind=self.engine)()

    def dump(self):
        for line in self.engine.raw_connection().iterdump():
            print line

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
                Hdu.product_lidvid == product_lidvid).where(
                Hdu.hdu_index == index)).scalar()

    def card_exists(self, keyword, hdu_index, product_lidvid):
        # type: (str, int, unicode) -> bool
        """
        Returns True iff there is a card with the given keyword in
        the n-th HDU of the FITS file for that product.
        """
        return self.session.query(
            exists().where(
                Card.product_lidvid == product_lidvid).where(
                Card.hdu_index == hdu_index).where(
                Card.keyword == keyword)).scalar()

    def create_bundle(self, bundle_lidvid):
        # type: (str) -> None
        """
        Create a bundle with this LIDVID if none exists.
        """

        LIDVID(bundle_lidvid)
        if not self.bundle_exists(bundle_lidvid):
            proposal_id = _lidvid_to_proposal_id(bundle_lidvid)
            self.session.add(Bundle(lidvid=bundle_lidvid,
                                    proposal_id=proposal_id))
            self.session.commit()

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
            instrument = _lidvid_to_instrument(collection_lidvid)
            suffix = _lidvid_to_suffix(collection_lidvid)
            self.session.add(
                NonDocumentCollection(
                    lidvid=collection_lidvid,
                    collection_lidvid=collection_lidvid,
                    bundle_lidvid=bundle_lidvid,
                    instrument=instrument,
                    suffix=suffix))
            self.session.commit()

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
            self.session.commit()

    def create_fits_file(self, basename, product_lidvid, hdu_count):
        # type: (unicode, str, int) -> None
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
                         product_lidvid=product_lidvid,
                         hdu_count=hdu_count))
            self.session.commit()
            assert self.fits_file_exists(basename, product_lidvid)

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
            self.session.commit()

    def get_bundle(self, lidvid):
        # type: (unicode) -> Bundle
        return self.session.query(Bundle).filter(
            Bundle.lidvid == lidvid).one()

    def get_collection(self, lidvid):
        # type: (unicode) -> Collection
        return self.session.query(Collection).filter(
            Collection.lidvid == lidvid).one()

    def get_product(self, lidvid):
        # type: (unicode) -> Product
        return self.session.query(Product).filter(
            Product.lidvid == lidvid).one()

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

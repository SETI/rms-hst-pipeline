import re

from sqlalchemy import create_engine, exists
from sqlalchemy.orm import sessionmaker
from typing import TYPE_CHECKING

import pdart.pds4.Bundle
import pdart.pds4.Collection
from pdart.new_db.SqlAlchTables import BadFitsFile, BrowseFile, \
    BrowseProduct, Bundle, BundleLabel, Card, Collection, \
    CollectionInventory, CollectionLabel, DocumentCollection, \
    DocumentFile, DocumentProduct, File, FitsFile, FitsProduct, Hdu, \
    NonDocumentCollection, Product, ProductLabel, ProposalInfo, create_tables
from pdart.new_db.Utils import file_md5
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID

if TYPE_CHECKING:
    from typing import Any, Dict, List, Tuple

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


def _lidvid_to_prefix(nondoc_collection_lidvid):
    lid = LIDVID(nondoc_collection_lidvid).lid()
    collection_id = lid.collection_id
    return re.match(pdart.pds4.Collection.Collection.DIRECTORY_PATTERN,
                    collection_id).group(2)


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

    ############################################################

    def create_bundle(self, bundle_lidvid):
        # type: (str) -> None
        """
        Create a bundle with this LIDVID if none exists.
        """

        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if not self.bundle_exists(bundle_lidvid):
            proposal_id = _lidvid_to_proposal_id(bundle_lidvid)
            self.session.add(Bundle(lidvid=bundle_lidvid,
                                    proposal_id=proposal_id))
            self.session.commit()

    def bundle_exists(self, bundle_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a bundle with the given LIDVID exists in the database.
        """
        return self.session.query(
            exists().where(Bundle.lidvid == bundle_lidvid)).scalar()

    def get_bundle(self):
        # type: () -> Bundle
        return self.session.query(Bundle).one()

    def get_bundle_collections(self, bundle_lidvid):
        # type: (str) -> List[Collection]
        return self.session.query(Collection).filter(
            Collection.bundle_lidvid == bundle_lidvid).all()

    ############################################################

    def create_document_collection(self, collection_lidvid, bundle_lidvid):
        # type: (str, str) -> None
        """
        Create a document_collection with this LIDVID if none exists.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.collection_exists(collection_lidvid):
            if self.document_collection_exists(collection_lidvid):
                pass
            else:
                raise Exception(
                    'non-document-collection with LIDVID %s already exists'
                    % collection_lidvid)
        else:
            self.session.add(
                DocumentCollection(
                    lidvid=collection_lidvid,
                    bundle_lidvid=bundle_lidvid))
            self.session.commit()

    def create_non_document_collection(self, collection_lidvid, bundle_lidvid):
        # type: (str, str) -> None
        """
        Create a non_document_collection with this LIDVID if none exists.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.collection_exists(collection_lidvid):
            if self.non_document_collection_exists(collection_lidvid):
                pass
            else:
                raise Exception(
                    'document-collection with LIDVID %s already exists'
                    % collection_lidvid)
        else:
            instrument = _lidvid_to_instrument(collection_lidvid)
            prefix = _lidvid_to_prefix(collection_lidvid)
            suffix = _lidvid_to_suffix(collection_lidvid)
            self.session.add(
                NonDocumentCollection(
                    lidvid=collection_lidvid,
                    collection_lidvid=collection_lidvid,
                    bundle_lidvid=bundle_lidvid,
                    instrument=instrument,
                    prefix=prefix,
                    suffix=suffix))
            self.session.commit()

    def collection_exists(self, collection_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a collection with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(Collection.lidvid == collection_lidvid)).scalar()

    def document_collection_exists(self, collection_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a document collection with the given LIDVID
        exists in the database.
        """
        return self.session.query(
            exists().where(
                DocumentCollection.collection_lidvid ==
                collection_lidvid)).scalar()

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

    def get_collection(self, lidvid):
        # type: (str) -> Collection
        return self.session.query(Collection).filter(
            Collection.lidvid == lidvid).one()

    def get_collection_products(self, collection_lidvid):
        # type: (str) -> List[Product]
        return self.session.query(Product).filter(
            Product.collection_lidvid == collection_lidvid).all()

    ############################################################

    def create_browse_product(self, browse_product_lidvid,
                              fits_product_lidvid, collection_lidvid):
        # type: (str, str, str) -> None
        """
        Create a product with this LIDVID if none exists.
        """
        assert LIDVID(browse_product_lidvid).is_product_lidvid()
        assert LIDVID(fits_product_lidvid).is_product_lidvid()
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.product_exists(fits_product_lidvid):
            if not self.fits_product_exists(fits_product_lidvid):
                raise Exception('product %s is not a FITS product' %
                                fits_product_lidvid)
        else:
            raise Exception('FITS product %s must exist before building '
                            'browse product %s' % (fits_product_lidvid,
                                                   browse_product_lidvid))

        if self.product_exists(browse_product_lidvid):
            if self.browse_product_exists(browse_product_lidvid):
                pass
            else:
                raise Exception(
                    'non-browse product with LIDVID %s already exists' %
                    browse_product_lidvid)
        else:
            self.session.add(
                BrowseProduct(lidvid=browse_product_lidvid,
                              collection_lidvid=collection_lidvid,
                              fits_product_lidvid=fits_product_lidvid))
            self.session.commit()

    def create_document_product(self, product_lidvid, collection_lidvid):
        # type: (str, str) -> None
        """
        Create a product with this LIDVID if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.product_exists(product_lidvid):
            if self.document_product_exists(product_lidvid):
                pass
            else:
                raise Exception(
                    'non-document product with LIDVID %s already exists' %
                    product_lidvid)
        else:
            self.session.add(
                DocumentProduct(lidvid=product_lidvid,
                                collection_lidvid=collection_lidvid))
            self.session.commit()

    def create_fits_product(self, product_lidvid, collection_lidvid):
        # type: (str, str) -> None
        """
        Create a product with this LIDVID if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        LIDVID(collection_lidvid)
        if self.product_exists(product_lidvid):
            if self.fits_product_exists(product_lidvid):
                pass
            else:
                raise Exception(
                    'non-FITS product with LIDVID %s already exists' %
                    LIDVID(product_lidvid))
        else:
            self.session.add(
                FitsProduct(lidvid=product_lidvid,
                            collection_lidvid=collection_lidvid))
            self.session.commit()

    def product_exists(self, product_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a product with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(Product.lidvid == product_lidvid)).scalar()

    def browse_product_exists(self, product_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a browse product with the given LIDVID exists
        in the database.
        """
        return self.session.query(
            exists().where(
                BrowseProduct.product_lidvid == product_lidvid)).scalar()

    def document_product_exists(self, product_lidvid):
        # type: (str) -> bool

        """
        Returns True iff a document product with the given LIDVID
        exists in the database.
        """
        return self.session.query(
            exists().where(
                DocumentProduct.product_lidvid ==
                product_lidvid)).scalar()

    def fits_product_exists(self, product_lidvid):
        # type: (str) -> bool
        """
        Returns True iff a FITS product with the given LIDVID exists in the
        database.
        """
        return self.session.query(
            exists().where(
                FitsProduct.product_lidvid == product_lidvid)).scalar()

    def get_product(self, lidvid):
        # type: (str) -> Product
        return self.session.query(Product).filter(
            Product.lidvid == lidvid).one()

    def get_product_file(self, product_lidvid):
        # type: (str) -> File
        """When you know there's only one, as in browse and FITS products"""
        return self.session.query(File).filter(
            File.product_lidvid == product_lidvid).one()

    def get_product_files(self, product_lidvid):
        # type: (str) -> List[File]
        return self.session.query(File).filter(
            File.product_lidvid == product_lidvid).all()

    ############################################################

    def create_bad_fits_file(self, os_filepath, basename, product_lidvid,
                             exception_message):
        # type: (unicode, unicode, str, str) -> None
        """
        Create a bad FITS file record with this basename belonging to
        the product if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.fits_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                BadFitsFile(basename=basename,
                            md5_hash=file_md5(os_filepath),
                            product_lidvid=product_lidvid,
                            exception_message=exception_message))
            self.session.commit()

    def create_browse_file(self, os_filepath, basename, product_lidvid,
                           byte_size):
        # type: (unicode, unicode, str, int) -> None
        """
        Create a browse file with this basename belonging to the product
        if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.browse_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                BrowseFile(basename=basename,
                           md5_hash=file_md5(os_filepath),
                           product_lidvid=product_lidvid,
                           byte_size=byte_size))
            self.session.commit()
            assert self.browse_file_exists(basename, product_lidvid)

    def create_document_file(self, os_filepath, basename, product_lidvid):
        # type: (unicode, unicode, str) -> None
        """
        Create a document file with this basename belonging to the product
        if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.document_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                DocumentFile(basename=basename,
                             md5_hash=file_md5(os_filepath),
                             product_lidvid=product_lidvid))
            self.session.commit()
            assert self.document_file_exists(basename, product_lidvid)

    def create_fits_file(self, os_filepath, basename, product_lidvid,
                         hdu_count):
        # type: (unicode, unicode, str, int) -> None
        """
        Create a FITS file with this basename belonging to the product
        if none exists.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.fits_file_exists(basename, product_lidvid):
            pass
        else:
            self.session.add(
                FitsFile(basename=basename,
                         md5_hash=file_md5(os_filepath),
                         product_lidvid=product_lidvid,
                         hdu_count=hdu_count))
            self.session.commit()
            assert self.fits_file_exists(basename, product_lidvid)

    def file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a file with the given LIDVID and basename
        exists in the database.
        """
        return self.session.query(
            exists().where(File.basename == basename).where(
                File.product_lidvid == product_lidvid)).scalar()

    def bad_fits_file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a bad FITS file record with the given LIDVID
        and basename exists in the database.
        """
        return self.session.query(
            exists().where(BadFitsFile.basename == basename).where(
                BadFitsFile.product_lidvid == product_lidvid).where(
                File.type == 'bad_fits_file')).scalar()

    def browse_file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a browse file with the given LIDVID and
        basename exists in the database.
        """
        return self.session.query(
            exists().where(BrowseFile.basename == basename).where(
                BrowseFile.product_lidvid == product_lidvid).where(
                File.type == 'browse_file')).scalar()

    def document_file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a document file with the given LIDVID and
        basename exists in the database.
        """
        return self.session.query(
            exists().where(DocumentFile.basename == basename).where(
                DocumentFile.product_lidvid == product_lidvid).where(
                File.type == 'document_file')).scalar()

    def fits_file_exists(self, basename, product_lidvid):
        # type: (unicode, str) -> bool
        """
        Returns True iff a FITS file with the given LIDVID and
        basename exists in the database.
        """
        return self.session.query(
            exists().where(FitsFile.basename == basename).where(
                FitsFile.product_lidvid == product_lidvid).where(
                File.type == 'fits_file')).scalar()

    def get_file(self, basename, product_lidvid):
        # type: (unicode, str) -> File
        return self.session.query(File).filter(
            File.product_lidvid == product_lidvid,
            File.basename == basename).one()

    ############################################################

    # The pattern of creation and access function used in higher-level
    # objects (bundles, collections, products) you see above break
    # down at this point, since queries inside the FITS file are
    # handled differently.

    def hdu_exists(self, index, basename, product_lidvid):
        # type: (int, unicode, str) -> bool
        """
        Returns True iff the n-th HDU for that FITS file exists
        """
        return self.session.query(
            exists().where(
                Hdu.product_lidvid == product_lidvid).where(
                Hdu.hdu_index == index)).scalar()

    def get_hdu(self, index, basename, product_lidvid):
        # type: (int, unicode, str) -> Hdu
        return self.session.query(Hdu).filter(
            Hdu.product_lidvid == product_lidvid,
            Hdu.basename == basename,
            Hdu.index == index).one()

    def get_file_offsets(self, fits_product_lidvid):
        # type: (unicode) -> List[Tuple[int, int, int, int]]
        hdus = self.session.query(Hdu).filter(
            Hdu.product_lidvid == fits_product_lidvid).order_by(
            Hdu.hdu_index)
        return [(hdu.hdu_index, hdu.hdr_loc, hdu.dat_loc, hdu.dat_span)
                for hdu in hdus]

    ############################################################

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

    def get_card_dictionaries(self, fits_product_lidvid, basename):
        # type: (str, unicode) -> List[Dict[str, Any]]
        """
        Return a list of dictionaries mapping FITS keys to their
        values, one per Hdu in the FITS file.
        """

        def get_card_dictionary(index):
            # type: (int) -> Dict[str, Any]
            cards = self.session.query(Card).filter(
                Card.product_lidvid == fits_product_lidvid).filter(
                Card.hdu_index == index)
            return {card.keyword: card.value for card in cards}

        file = self.get_file(basename, fits_product_lidvid)
        return [get_card_dictionary(i) for i in range(file.hdu_count)]

    ############################################################

    def create_bundle_label(self, os_filepath, basename, bundle_lidvid):
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        if self.bundle_label_exists(bundle_lidvid):
            pass
        else:
            self.session.add(
                BundleLabel(bundle_lidvid=bundle_lidvid,
                            basename=basename,
                            md5_hash=file_md5(os_filepath)))
            self.session.commit()
            assert self.bundle_label_exists(bundle_lidvid)

    def bundle_label_exists(self, bundle_lidvid):
        # type: (str) -> bool
        """
        Returns True iff there is a label for the bundle with the
        given LIDVID.
        """
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        return self.session.query(
            exists().where(
                BundleLabel.bundle_lidvid == bundle_lidvid)).scalar()

    def get_bundle_label(self, bundle_lidvid):
        # type: (str) -> BundleLabel
        """
        Returns the label for the bundle with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(bundle_lidvid).is_bundle_lidvid()
        return self.session.query(BundleLabel).filter(
            BundleLabel.bundle_lidvid == bundle_lidvid).one()

    ############################################################

    def create_collection_label(self, os_filepath,
                                basename, collection_lidvid):
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.collection_label_exists(collection_lidvid):
            pass
        else:
            self.session.add(
                CollectionLabel(collection_lidvid=collection_lidvid,
                                basename=basename,
                                md5_hash=file_md5(os_filepath)))
            self.session.commit()
            assert self.collection_label_exists(collection_lidvid)

    def collection_label_exists(self, collection_lidvid):
        # type: (str) -> bool
        """
        Returns True iff there is a label for the collection with the
        given LIDVID.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return self.session.query(
            exists().where(
                CollectionLabel.collection_lidvid == collection_lidvid
            )).scalar()

    def get_collection_label(self, collection_lidvid):
        # type: (str) -> CollectionLabel
        """
        Returns the label for the collection with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return self.session.query(CollectionLabel).filter(
            CollectionLabel.collection_lidvid == collection_lidvid).one()

    ############################################################

    def create_collection_inventory(self, os_filepath,
                                    basename, collection_lidvid):
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        if self.collection_inventory_exists(collection_lidvid):
            pass
        else:
            self.session.add(
                CollectionInventory(collection_lidvid=collection_lidvid,
                                    basename=basename,
                                    md5_hash=file_md5(os_filepath)))
            self.session.commit()
            assert self.collection_inventory_exists(collection_lidvid)

    def collection_inventory_exists(self, collection_lidvid):
        # type: (str) -> bool
        """
        Returns True iff there is a inventory for the collection with the
        given LIDVID.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return self.session.query(
            exists().where(
                CollectionInventory.collection_lidvid == collection_lidvid
            )).scalar()

    def get_collection_inventory(self, collection_lidvid):
        # type: (str) -> CollectionInventory
        """
        Returns the inventory for the collection with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(collection_lidvid).is_collection_lidvid()
        return self.session.query(CollectionInventory).filter(
            CollectionInventory.collection_lidvid == collection_lidvid).one()

    ############################################################

    def create_product_label(self, os_filepath,
                             basename, product_lidvid):
        assert LIDVID(product_lidvid).is_product_lidvid()
        if self.product_label_exists(product_lidvid):
            pass
        else:
            self.session.add(
                ProductLabel(product_lidvid=product_lidvid,
                             basename=basename,
                             md5_hash=file_md5(os_filepath)))
            self.session.commit()
            assert self.product_label_exists(product_lidvid)

    def product_label_exists(self, product_lidvid):
        # type: (str) -> bool
        """
        Returns True iff there is a label for the product with the
        given LIDVID.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        return self.session.query(
            exists().where(
                ProductLabel.product_lidvid == product_lidvid
            )).scalar()

    def get_product_label(self, product_lidvid):
        # type: (str) -> ProductLabel
        """
        Returns the label for the product with the given LIDVID, or
        raises an exception.
        """
        assert LIDVID(product_lidvid).is_product_lidvid()
        return self.session.query(ProductLabel).filter(
            ProductLabel.product_lidvid == product_lidvid).one()

    ############################################################

    def proposal_info_exists(self, bundle_lid):
        # type: (str) -> bool
        assert LID(bundle_lid).is_bundle_lid()
        return self.session.query(
            exists().where(ProposalInfo.bundle_lid == bundle_lid)).scalar()

    def create_proposal_info(self, bundle_lid,
                             proposal_title, pi_name, author_list,
                             proposal_year, publication_year):
        # type: (str, unicode, unicode, unicode, unicode, unicode) -> None
        '''
        Creates a record of proposal info for the given LID.

        NOTE: We don't allow updating through this interface now.  We
        might want to allow it in the future.
        '''
        assert LID(bundle_lid).is_bundle_lid()
        if self.proposal_info_exists(bundle_lid):
            raise Exception('proposal info with LID %s already exists' %
                            bundle_lid)
        else:
            self.session.add(ProposalInfo(
                    bundle_lid=bundle_lid,
                    proposal_title=proposal_title,
                    pi_name=pi_name,
                    author_list=author_list,
                    proposal_year=proposal_year,
                    publication_year=publication_year))

    def get_proposal_info(self, bundle_lid):
        # type: (str) -> ProposalInfo
        return self.session.query(ProposalInfo).filter(
            ProposalInfo.bundle_lid == bundle_lid).one()

    ############################################################

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

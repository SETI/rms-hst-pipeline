"""
Tables used by SqlAlchemy to build a database.  This will replace the
previous implementation which used raw SQL calls.
"""
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship, sessionmaker
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import *
    from sqlalchemy.orm import Session
    from sqlalchemy.schema import *
    from sqlalchemy.types import *

    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P


Base = declarative_base()
# type: Any


class Bundle(Base):
    """
    A database representation of a PDS4 bundle.
    """
    __tablename__ = 'bundles'

    lid = Column(String, primary_key=True, nullable=False)
    proposal_id = Column(Integer, nullable=False)
    archive_path = Column(String, nullable=False)
    full_filepath = Column(String, nullable=False)
    label_filepath = Column(String, nullable=False)
    is_complete = Column(Boolean, nullable=False)

    def __repr__(self):
        return 'Bundle(lid={0.lid:s},is_complete={0.is_complete})'.format(
            self)


class BadFitsFile(Base):
    """
    A database entry for a FITS file that cannot be read without
    errors.
    """
    __tablename__ = 'bad_fits_files'

    lid = Column(String, primary_key=True, nullable=False)
    filepath = Column(String, nullable=False)
    message = Column(String, nullable=False)

    def __repr__(self):
        return 'BadFitsFile(lid=%s)' % self.lid


class Collection(Base):
    """
    A database representation of a PDS4 collection.
    """
    __tablename__ = 'collections'

    lid = Column(String, primary_key=True, nullable=False)
    bundle_lid = Column(String, ForeignKey('bundles.lid'),
                        nullable=False, index=True)
    full_filepath = Column(String, nullable=False)
    label_filepath = Column(String, nullable=False)
    inventory_name = Column(String, nullable=False)
    inventory_filepath = Column(String, nullable=False)
    type = Column(String(24), nullable=False)

    bundle = relationship('Bundle', backref=backref('collections',
                                                    order_by=lid))

    __mapper_args__ = {
        'polymorphic_identity': 'collection',
        'polymorphic_on': type
        }


class DocumentCollection(Collection):
    """
    A database representation of a PDS4 collection containing
    documents.
    """
    __tablename__ = 'document_collections'
    collection_lid = Column(String, ForeignKey('collections.lid'),
                            primary_key=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'document_collection',
        }

    def __repr__(self):
        return 'DocumentCollection(lid=%s)' % self.lid


class NonDocumentCollection(Collection):
    """
    A database representation of a PDS4 collection which does not
    contain documents.
    """
    __tablename__ = 'non_document_collections'
    collection_lid = Column(String, ForeignKey('collections.lid'),
                            primary_key=True, nullable=False)
    prefix = Column(String, nullable=False)
    suffix = Column(String, nullable=False)
    instrument = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'non_document_collection',
        }

    def __repr__(self):
        return 'NonDocumentCollection(lid=%s)' % self.lid


Index('idx_non_document_collections_prefix_suffix',
      NonDocumentCollection.prefix,
      NonDocumentCollection.suffix)


class Product(Base):
    """
    A database representation of a PDS4 product.
    """
    __tablename__ = 'products'

    lid = Column(String, primary_key=True, nullable=False)
    collection_lid = Column(String, ForeignKey('collections.lid'),
                            nullable=False, index=True)
    label_filepath = Column(String, nullable=False)
    type = Column(String(16), nullable=False)

    collection = relationship('Collection', backref=backref('products',
                                                            order_by=lid))

    __mapper_args__ = {
        'polymorphic_identity': 'product',
        'polymorphic_on': type
        }


class FitsProduct(Product):
    """
    A database representation of a PDS4 observational product
    consisting of a single FITS file.
    """
    __tablename__ = 'fits_products'
    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False)
    fits_filepath = Column(String, nullable=False)
    visit = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'fits_product',
        }

    def __repr__(self):
        return 'FitsProduct(lid=%s, fits_filepath=%s)' % (self.lid,
                                                          self.fits_filepath)

    # full_filepath, filename, hdu_count


class BrowseProduct(Product):
    """
    A database representation of a PDS4 product consisting of browse
    images.
    """
    __tablename__ = 'browse_products'
    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False)
    browse_filepath = Column(String, nullable=False)
    object_length = Column(Integer, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'browse_product',
        }

    def __repr__(self):
        return 'BrowseProduct(lid=%s, fits_filepath=%s)' % \
            (self.lid, self.browse_filepath)


class DocumentProduct(Product):
    """
    A database representation of a PDS4 product consisting of
    documents.
    """
    __tablename__ = 'document_products'
    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False)
    document_filepath = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'document_product',
        }

    def __repr__(self):
        return 'DocumentProduct(lid=%s)' % \
            (self.lid)


class DocumentFile(Base):
    """
    A database representation of a single document file, part of a
    document product.
    """
    __tablename__ = 'document_files'
    product_lid = Column(String,
                         ForeignKey('document_products.product_lid'),
                         primary_key=True, nullable=False)
    file_basename = Column(String, primary_key=True, nullable=False)

    bundle = relationship('DocumentProduct',
                          backref=backref('document_files',
                                          order_by=file_basename))


class Hdu(Base):
    """
    A database representation of a FITS HDU.
    """
    __tablename__ = 'hdus'

    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False, index=True)
    hdu_index = Column(Integer, primary_key=True, nullable=False)
    """The zero-based index of this HDU within its product's FITS file."""
    hdr_loc = Column(Integer, nullable=False)
    """The starting byte location of the header in the file"""
    dat_loc = Column(Integer, nullable=False)
    """The starting byte location of the data block in the file"""
    dat_span = Column(Integer, nullable=False)
    """The data size including padding"""

    product = relationship('FitsProduct', backref=backref('hdus',
                                                          order_by=hdu_index))

    def __repr__(self):
        return 'Hdu(product_lid=%s, hdu_index=%d)' % \
            (self.product_lid, self.hdu_index)


class Card(Base):
    """
    A database representation of a card within an HDU of a FITS file.
    """
    __tablename__ = 'cards'

    id = Column(Integer, primary_key=True, nullable=False)
    product_lid = Column(String, ForeignKey('products.lid'), nullable=False)
    hdu_index = Column(Integer, ForeignKey('hdus.hdu_index'), nullable=False)
    keyword = Column(String, nullable=False)
    value = Column(String, nullable=True)

    hdu = relationship('Hdu', backref=backref('cards',
                                              order_by=id))

    def __repr__(self):
        return 'Card(product_lid=%s, hdu_index=%d, keyword=%s, value=%s)' % \
            (self.product_lid, self.hdu_index, self.keyword, self.value)


Index('idx_cards_product_hdu_index', Card.product_lid, Card.hdu_index)


def lookup_card(hdu, keyword):
    """
    Look up a card by keyword within an HDU's list of cards.  TODO One
    should be able to implement this using bracket notation, but I
    haven't figured out how yet.
    """
    matching_cards = [card for card in hdu.cards if card.keyword == keyword]
    if matching_cards:
        return '' + matching_cards[0].value
    else:
        return None


def _create_database_tables(db_filepath):
    # type: (unicode) -> Engine
    """
    Given a filepath for a database, create a new database file there
    containing the tables defined in this file and return the database
    engine.
    """
    engine = create_engine('sqlite:///' + db_filepath)
    Base.metadata.create_all(engine)
    return engine


def create_database_tables_and_session(db_filepath):
    # type: (unicode) -> Session
    """
    Given a filepath for a database, create a new database file there
    containing the tables defined in this file, and return a session
    of the database engine operating on that database.
    """
    engine = _create_database_tables(db_filepath)
    return sessionmaker(bind=engine)()


def db_bundle_exists(session, bundle):
    # type: (Session, B.Bundle) -> bool
    res = session.query(Bundle).filter_by(lid=str(bundle.lid)).one_or_none()
    return res is not None


def db_collection_exists(session, collection):
    # type: (Session, C.Collection) -> bool
    res = session.query(Collection).filter_by(
        lid=str(collection.lid)).one_or_none()
    return res is not None


def db_document_collection_exists(session, collection):
    # type: (Session, C.Collection) -> bool
    res = session.query(DocumentCollection).filter_by(
        collection_lid=str(collection.lid)).one_or_none()
    return res is not None


def db_non_document_collection_exists(session, collection):
    # type: (Session, C.Collection) -> bool
    res = session.query(NonDocumentCollection).filter_by(
        collection_lid=str(collection.lid)).one_or_none()
    return res is not None


def db_product_exists(session, product):
    # type: (Session, P.Product) -> bool
    res = session.query(Product).filter_by(lid=str(product.lid)).one_or_none()
    return res is not None


def db_fits_product_exists(session, product):
    # type: (Session, P.Product) -> bool
    res = session.query(FitsProduct).filter_by(
        product_lid=str(product.lid)).one_or_none()
    return res is not None


def db_browse_product_exists(session, product):
    # type: (Session, P.Product) -> bool
    res = session.query(BrowseProduct).filter_by(
        product_lid=str(product.lid)).one_or_none()
    return res is not None


def db_document_product_exists(session, product):
    # type: (Session, P.Product) -> bool
    res = session.query(DocumentProduct).filter_by(
        product_lid=str(product.lid)).one_or_none()
    return res is not None


def db_bad_fits_file_exists(session, product):
    # type: (Session, P.Product) -> bool
    res = session.query(BadFitsFile).filter_by(
        lid=str(product.lid)).one_or_none()
    return res is not None


if __name__ == '__main__':
    db_fp = ':memory:'
    eng = create_engine('sqlite:///' + db_fp, echo=True)
    Base.metadata.create_all(eng)

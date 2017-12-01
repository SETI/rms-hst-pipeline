import abc
from sqlalchemy import Column, ForeignKey, Index, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from sqlalchemy.engine import Engine

Base = declarative_base()  # type: Any


def create_tables(engine):
    # type: (Engine) -> None
    Base.metadata.create_all(engine)


class Bundle(Base):
    __tablename__ = 'bundles'
    lidvid = Column(String, primary_key=True, nullable=False)


############################################################

class Collection(Base):
    __tablename__ = 'collections'
    lidvid = Column(String, primary_key=True, nullable=False)
    bundle_lidvid = Column(String, ForeignKey('bundles.lidvid'),
                           nullable=False, index=True)
    type = Column(String(24), nullable=False)
    __mapper_args__ = {
        'polymorphic_identity': 'collection',
        'polymorphic_on': type
    }


class DocumentCollection(Collection):
    __tablename__ = 'document_collections'
    collection_lidvid = Column(String, ForeignKey('collections.lidvid'),
                               primary_key=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'document_collection',
    }


class NonDocumentCollection(Collection):
    __tablename__ = 'non_document_collections'
    collection_lidvid = Column(String, ForeignKey('collections.lidvid'),
                               primary_key=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'non_document_collection',
    }


############################################################

class Product(Base):
    __tablename__ = 'products'
    lidvid = Column(String, primary_key=True, nullable=False)
    collection_lidvid = Column(String, ForeignKey('collections.lidvid'),
                               nullable=False, index=True)
    type = Column(String(16), nullable=False)

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
    product_lidvid = Column(String, ForeignKey('products.lidvid'),
                            primary_key=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'fits_product',
    }


class BrowseProduct(Product):
    """
    A database representation of a PDS4 product consisting of browse
    images.
    """
    __tablename__ = 'browse_products'
    product_lidvid = Column(String, ForeignKey('products.lidvid'),
                            primary_key=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'browse_product',
    }


class DocumentProduct(Product):
    """
    A database representation of a PDS4 product consisting of
    documents.
    """
    __tablename__ = 'document_products'
    product_lidvid = Column(String, ForeignKey('products.lidvid'),
                            primary_key=True, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'document_product',
    }


############################################################

class DocumentFile(Base):
    """
    A database representation of a single document file, part of a
    document product.
    """
    __tablename__ = 'document_files'
    product_lidvid = Column(String,
                            ForeignKey('document_products.product_lidvid'),
                            primary_key=True, nullable=False)


############################################################

class Hdu(Base):
    """
    A database representation of a FITS HDU.
    """
    __tablename__ = 'hdus'

    product_lidvid = Column(String, ForeignKey('products.lidvid'),
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


class Card(Base):
    """
    A database representation of a card within an HDU of a FITS file.
    """
    __tablename__ = 'cards'

    id = Column(Integer, primary_key=True, nullable=False)
    product_lidvid = Column(String, ForeignKey('products.lidvid'),
                            nullable=False)
    hdu_index = Column(Integer, ForeignKey('hdus.hdu_index'), nullable=False)
    keyword = Column(String, nullable=False)
    value = Column(String, nullable=True)

    hdu = relationship('Hdu', backref=backref('cards',
                                              order_by=id))

    def __repr__(self):
        return 'Card(product_lid=%s, hdu_index=%d, keyword=%s, value=%s)' % \
               (self.product_lid, self.hdu_index, self.keyword, self.value)


Index('idx_cards_product_hdu_index', Card.product_lidvid, Card.hdu_index)

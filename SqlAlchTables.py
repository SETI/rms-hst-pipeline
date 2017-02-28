from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.engine import *
    from sqlalchemy.schema import *
    from sqlalchemy.types import *

Base = declarative_base()
# type: Any


class Bundle(Base):
    __tablename__ = 'bundles'

    lid = Column(String, primary_key=True, nullable=False)
    proposal_id = Column(Integer, nullable=False)
    archive_path = Column(String, nullable=False)
    full_filepath = Column(String, nullable=False)
    label_filepath = Column(String, nullable=False)


class BadFitsFile(Base):
    __tablename__ = 'bad_fits_files'

    lid = Column(String, primary_key=True, nullable=False)
    filepath = Column(String, nullable=False)
    message = Column(String, nullable=False)


class Collection(Base):
    __tablename__ = 'collections'

    lid = Column(String, primary_key=True, nullable=False)
    bundle_lid = Column(String, ForeignKey('bundles.lid'),
                        nullable=False, index=True)
    bundle = relationship('Bundle', backref=backref('collections',
                                                    order_by=lid))
    prefix = Column(String, nullable=False)
    suffix = Column(String, nullable=False)
    instrument = Column(String, nullable=False)
    full_filepath = Column(String, nullable=False)
    label_filepath = Column(String, nullable=False)
    inventory_name = Column(String, nullable=False)
    inventory_filepath = Column(String, nullable=False)

Index('idx_collections_prefix_suffix', Collection.prefix, Collection.suffix)


class Product(Base):
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
    __tablename__ = 'fits_products'
    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False)
    fits_filepath = Column(String, nullable=False)
    visit = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'fits_product',
        }
    # full_filepath, filename, hdu_count


class BrowseProduct(Product):
    __tablename__ = 'browse_products'
    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False)
    browse_filepath = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'browse_product',
        }


class DocumentProduct(Product):
    __tablename__ = 'document_products'
    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False)
    document_filepath = Column(String, nullable=False)

    __mapper_args__ = {
        'polymorphic_identity': 'document_product',
        }


class Hdu(Base):
    __tablename__ = 'hdus'

    product_lid = Column(String, ForeignKey('products.lid'),
                         primary_key=True, nullable=False, index=True)
    hdu_index = Column(Integer, primary_key=True, nullable=False)
    hdr_loc = Column(Integer, nullable=False)
    dat_loc = Column(Integer, nullable=False)
    dat_span = Column(Integer, nullable=False)
    product = relationship('FitsProduct', backref=backref('hdus',
                                                          order_by=hdu_index))


class Card(Base):
    __tablename__ = 'cards'

    id = Column(Integer, primary_key=True, nullable=False)
    product_lid = Column(String, ForeignKey('products.lid'), nullable=False)
    hdu_index = Column(Integer, ForeignKey('hdus.hdu_index'), nullable=False)
    keyword = Column(String, nullable=False)
    value = Column(String, nullable=True)
    hdu = relationship('Hdu', backref=backref('cards',
                                              order_by=id))

Index('idx_cards_product_hdu_index', Card.product_lid, Card.hdu_index)


def lookup_card(hdu, keyword):
    matching_cards = [card for card in hdu.cards if card.keyword == keyword]
    if matching_cards:
        return '' + matching_cards[0].value
    else:
        return None


if __name__ == '__main__':
    db_fp = ':memory:'
    eng = create_engine('sqlite:///' + db_fp, echo=True)
    Base.metadata.create_all(eng)

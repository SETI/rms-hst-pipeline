from contextlib import closing
import os
import os.path
import pyfits
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import *

from pdart.pds4.Archives import get_any_archive

Base = declarative_base()

class Bundle(Base):
    __tablename__ = 'bundle'

    lid = Column(String, primary_key=True)
    
class BadFitsFile(Base):
    __tablename__ = 'bad_fits_file'

    lid = Column(String, primary_key=True)
    filepath = Column(String)
    message = Column(String)
    
class Collection(Base):
    __tablename__ = 'collection'

    lid = Column(String, primary_key=True)
    bundle = Column(String, ForeignKey('bundle.lid'))
    idx_collection_bundle = Index('bundle')

class Product(Base):
    __tablename__ = 'product'

    lid = Column(String, primary_key=True)
    collection = Column(String, ForeignKey('collection.lid'))
    type = Column(String(16))

    __mapper_args__ = {
        'polymorphic_identity': 'product',
        'polymorphic_on': type
        }

class FitsProduct(Product):
    __tablename__ = 'fits_product'
    lid = Column(String, ForeignKey('product.lid'), primary_key=True)
    fits_filepath = Column(String)

    __mapper_args__ = {
        'polymorphic_identity': 'fits_product',
        }
    
class Hdu(Base):
    __tablename__ = 'hdu'

    product = Column(String, ForeignKey('product.lid'), primary_key=True)
    hdu_index = Column(Integer, primary_key=True)
    hdr_loc = Column(Integer)
    dat_loc = Column(Integer)
    dat_span = Column(Integer)

def handle_undefined(val):
    """Convert undefined values to None"""
    if isinstance(val, pyfits.card.Undefined):
        return None
    else:
        return val

class Card(Base):
    __tablename__ = 'card'

    id = Column(Integer, primary_key=True)
    product = Column(String, ForeignKey('product.lid'))
    hdu_index = Column(Integer)
    keyword = Column(String)
    value = Column(String)
    

class BrowseProduct(Product):
    __tablename__ = 'browse_product'
    lid = Column(String, ForeignKey('product.lid'), primary_key=True)
    browse_filepath = Column(String)

    __mapper_args__ = {
        'polymorphic_identity': 'browse_product',
        }
    
_NEW_DATABASE_NAME = 'sqlalch-database.db'
# type: str


def bundle_database_filepath(bundle):
    # type: (Bundle) -> unicode
    return os.path.join(bundle.absolute_filepath(), _NEW_DATABASE_NAME)


def open_bundle_database(bundle):
    # type: (Bundle) -> sqlite3.Connection
    return sqlite3.connect(bundle_database_filepath(bundle))


def run():
    archive = get_any_archive()
    for bundle in archive.bundles():
        db_fp = bundle_database_filepath(bundle)
        try:
            os.remove(db_fp)
        except OSError:
            pass
        engine = create_engine('sqlite:///' + db_fp)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        db_bundle = Bundle(lid=str(bundle.lid))
        session.add(db_bundle)
        session.commit()
        for collection in bundle.collections():
            db_collection = Collection(lid=str(collection.lid),
                                       bundle=str(bundle.lid))
            session.add(db_collection)
            session.commit()
            if collection.prefix() == 'data':
                for product in collection.products():
                        file = list(product.files())[0]
                        try:
                            with closing(pyfits.open(file.full_filepath())) as fits:
                                db_fits_product = FitsProduct(lid=str(product.lid),
                                                              collection=str(collection.lid),
                                                              fits_filepath=file.full_filepath())
                                for (n, hdu) in enumerate(fits):
                                    fileinfo = hdu.fileinfo()
                                    db_hdu = Hdu(product=str(product.lid),
                                                 hdu_index=n,
                                                 hdr_loc=fileinfo['hdrLoc'],
                                                 dat_loc=fileinfo['datLoc'],
                                                 dat_span=fileinfo['datSpan'])
                                    session.add(db_hdu)
                                    header = hdu.header
                                    for card in header.cards:
                                        if card.keyword:
                                            db_card = Card(product=str(product.lid),
                                                           hdu_index=n,
                                                           keyword=card.keyword,
                                                           value=handle_undefined(card.value))
                                            session.add(db_card)
                                            session.commit()
                                session.add(db_fits_product)
                                session.commit()
                        except IOError as e:
                            db_bad_fits_file = BadFitsFile(lid=str(product.lid),
                                                           filepath=file.full_filepath(),
                                                           message=str(e))
                            session.add(db_bad_fits_file)
                        session.commit()
        print db_fp

if __name__ == '__main__':
    run()

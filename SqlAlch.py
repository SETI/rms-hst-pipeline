from contextlib import closing
import os
import os.path
import pyfits
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import *
import sys

from pdart.pds4.Archives import get_any_archive
from pdart.xml.Schema import verify_label_or_raise_fp
import SqlAlchLabels
from SqlAlchTables import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3


def handle_undefined(val):
    """Convert undefined values to None"""
    if isinstance(val, pyfits.card.Undefined):
        return None
    else:
        return val


_NEW_DATABASE_NAME = 'sqlalch-database.db'
# type: str

VERIFY = False
# type: bool


def bundle_database_filepath(bundle):
    # type: (Bundle) -> unicode
    return os.path.join(bundle.absolute_filepath(), _NEW_DATABASE_NAME)


def open_bundle_database(bundle):
    # type: (Bundle) -> sqlite3.Connection
    return sqlite3.connect(bundle_database_filepath(bundle))


def add_cards(session, product_lid, hdu_index, header):
    # type: (Session, unicode, int, Any) -> None
    cards = [Card(product_lid=product_lid,
                  hdu_index=hdu_index,
                  keyword=card.keyword,
                  value=handle_undefined(card.value))
             for card in header.cards if card.keyword]
    session.bulk_save_objects(cards)


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
        db_bundle = Bundle(lid=str(bundle.lid),
                           proposal_id=bundle.proposal_id(),
                           archive_path=os.path.abspath(archive.root),
                           full_filepath=bundle.absolute_filepath(),
                           label_filepath=bundle.label_filepath())
        session.add(db_bundle)
        session.commit()
        for collection in bundle.collections():
            db_collection = Collection(
                lid=str(collection.lid),
                bundle_lid=str(bundle.lid),
                prefix=collection.prefix(),
                suffix=collection.suffix(),
                instrument=collection.instrument(),
                full_filepath=collection.absolute_filepath(),
                label_filepath=collection.label_filepath(),
                inventory_name=collection.inventory_name(),
                inventory_filepath=collection.inventory_filepath())
            session.add(db_collection)
            session.commit()
            if collection.prefix() == 'data':
                for product in collection.products():
                    db_fits_product = None
                    print '    ', product.lid
                    file = list(product.files())[0]
                    try:
                        with closing(pyfits.open(
                                file.full_filepath())) as fits:
                            db_fits_product = FitsProduct(
                                lid=str(product.lid),
                                collection_lid=str(collection.lid),
                                fits_filepath=file.full_filepath(),
                                label_filepath=product.label_filepath(),
                                visit=product.visit())
                            for (n, hdu) in enumerate(fits):
                                fileinfo = hdu.fileinfo()
                                db_hdu = Hdu(product_lid=str(product.lid),
                                             hdu_index=n,
                                             hdr_loc=fileinfo['hdrLoc'],
                                             dat_loc=fileinfo['datLoc'],
                                             dat_span=fileinfo['datSpan'])
                                session.add(db_hdu)
                                add_cards(session,
                                          str(product.lid),
                                          n,
                                          hdu.header)
                            session.add(db_fits_product)
                    except IOError as e:
                        db_bad_fits_file = BadFitsFile(
                            lid=str(product.lid),
                            filepath=file.full_filepath(),
                            message=str(e))
                        session.add(db_bad_fits_file)
                    session.commit()

                    if db_fits_product:
                        label = SqlAlchLabels.make_product_observational_label(
                            db_fits_product)
                        label_filepath = db_fits_product.label_filepath
                        with open(label_filepath, 'w') as f:
                            f.write(label)
                        try:
                            if VERIFY:
                                verify_label_or_raise_fp(label_filepath)
                            print '    ', 'label:', \
                                os.path.relpath(label_filepath, archive.root)
                        except:
                            print '#### failed on', label_filepath
                            raise
        print db_fp

if __name__ == '__main__':
    run()

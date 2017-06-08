from contextlib import closing
import logging
import os
import os.path
import pyfits
import sys

from pdart.db.SqlAlchDBName import DATABASE_NAME
from pdart.db.SqlAlchTables import *
from pdart.db.SqlAlchUtils import bundle_database_filepath
from pdart.pds4.Archives import get_any_archive
from pdart.pds4labels.SqlAlchLabels import make_product_observational_label
from pdart.xml.Schema import verify_label_or_raise_fp

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    import sqlite3
    import pdart.pds4.Archive as A
    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P


def handle_undefined(val):
    """Convert undefined values to None"""
    if isinstance(val, pyfits.card.Undefined):
        return None
    else:
        return val


VERIFY = False
# type: bool


def db_add_cards(session, product_lid, hdu_index, header):
    # type: (Session, unicode, int, Any) -> None
    cards = [Card(product_lid=product_lid,
                  hdu_index=hdu_index,
                  keyword=card.keyword,
                  value=handle_undefined(card.value))
             for card in header.cards if card.keyword]
    session.bulk_save_objects(cards)


def db_add_bundle(archive, bundle):
    # type: (A.Archive, B.Bundle) -> None
    db_fp = bundle_database_filepath(bundle)
    try:
        os.remove(db_fp)
    except OSError:
        pass
    session = create_database_tables_and_session(db_fp)
    db_bundle = Bundle(lid=str(bundle.lid),
                       proposal_id=bundle.proposal_id(),
                       archive_path=os.path.abspath(archive.root),
                       full_filepath=bundle.absolute_filepath(),
                       label_filepath=bundle.label_filepath(),
                       is_complete=False)
    session.add(db_bundle)
    session.commit()
    for collection in bundle.collections():
        if collection.lid.collection_id == 'document':
            db_add_document_collection(session, archive, bundle, collection)
        else:
            db_add_non_document_collection(session, archive,
                                           bundle, collection)
    logging.getLogger(__name__).info(db_fp)


def db_add_document_collection(session, archive, bundle, collection):
    # type: (Session, A.Archive, B.Bundle, C.Collection) -> None
    """
    Add a DocumentCollection entry into the database for the collection.
    """
    # PRECONDITION
    assert db_bundle_exists(session, bundle)

    db_collection = DocumentCollection(
        lid=str(collection.lid),
        bundle_lid=str(bundle.lid),
        full_filepath=collection.absolute_filepath(),
        label_filepath=collection.label_filepath(),
        inventory_name=collection.inventory_name(),
        inventory_filepath=collection.inventory_filepath())
    session.add(db_collection)
    session.commit()
    # POSTCONDITION
    assert db_document_collection_exists(session, collection)


def db_add_non_document_collection(session, archive, bundle, collection):
    # type: (Session, A.Archive, B.Bundle, C.Collection) -> Collection
    """
    Add a NonDocumentCollection entry into the database for the
    collection.
    """
    # PRECONDITION
    assert db_bundle_exists(session, bundle)

    db_collection = NonDocumentCollection(
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
    # if collection.prefix() == 'data':
    #     for product in collection.products():
    #        db_add_product(session, archive, collection, product)

    # POSTCONDITION
    assert db_non_document_collection_exists(session, collection)
    return db_collection


def db_add_product(session, archive, collection, product):
    # type: (Session, A.Archive, C.Collection, P.Product) -> Product
    """
    Add a FitsProduct entry into the database for the Product; if the
    file cannot be parsed, add a BadFitsFile entry instead.
    """
    # PRECONDITIONS
    assert db_non_document_collection_exists(session, collection)
    # and the FITS file exists in the filesystem
    fits_filepath = product.first_filepath()
    assert os.path.isfile(fits_filepath)

    logging.getLogger(__name__).info(str(product.lid))
    db_fits_product = None
    try:
        with closing(pyfits.open(
                fits_filepath)) as fits:
            db_fits_product = FitsProduct(
                lid=str(product.lid),
                collection_lid=str(collection.lid),
                fits_filepath=fits_filepath,
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
                db_add_cards(session,
                             str(product.lid),
                             n,
                             hdu.header)
            session.add(db_fits_product)
    except IOError as e:
        db_bad_fits_file = BadFitsFile(
            lid=str(product.lid),
            filepath=fits_filepath,
            message=str(e))
        session.add(db_bad_fits_file)
        logging.getLogger(__name__).error(
            'ERROR: bad FITS file', str(product.lid))
    session.commit()

    if db_fits_product:
        label = make_product_observational_label(db_fits_product)
        label_filepath = str(db_fits_product.label_filepath)
        with open(label_filepath, 'w') as f:
            f.write(label)
        try:
            if VERIFY:
                verify_label_or_raise_fp(label_filepath)
            logging.getLogger(__name__).info(
                'label: %s' % os.path.relpath(label_filepath, archive.root))
        except:
            logging.getLogger(__name__).error(
                '#### failed on %s' % label_filepath)
            raise

    # POSTCONDITION
    assert db_fits_product_exists(session, product) or \
        db_bad_fits_file_exists(session, product)

    return db_fits_product


def run():
    archive = get_any_archive()
    for bundle in archive.bundles():
        db_add_bundle(archive, bundle)

if __name__ == '__main__':
    run()

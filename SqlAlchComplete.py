import os
import os.path
import shutil

from pdart.db.SqlAlchDBName import DATABASE_NAME
from pdart.db.SqlAlchDocs import db_add_document_collection, \
    db_add_document_product, populate_document_collection
from pdart.db.SqlAlchTables import Base, Bundle, \
    create_database_tables_and_session, BrowseProduct, DocumentCollection,\
    DocumentProduct, FitsProduct, NonDocumentCollection
from pdart.pds4.Archives import get_any_archive
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.pds4labels.SqlAlchLabels import make_browse_product, \
    make_db_browse_product, make_and_save_product_browse_label, \
    make_and_save_product_bundle_label, \
    make_and_save_product_collection_label, \
    make_and_save_product_document_label, \
    make_and_save_product_observational_label
from pdart.xml.Schema import verify_label_or_raise

from SqlAlch import db_add_product, db_add_non_document_collection

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    import pdart.pds4.Archive as A
    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P
    _NDCollection = NonDocumentCollection


def db_add_bundle(session, archive, bundle):
    # type: (Session, A.Archive, B.Bundle) -> Bundle
    db_bundle = Bundle(lid=str(bundle.lid),
                       proposal_id=bundle.proposal_id(),
                       archive_path=os.path.abspath(archive.root),
                       full_filepath=bundle.absolute_filepath(),
                       label_filepath=bundle.label_filepath())
    session.add(db_bundle)
    session.commit()
    return db_bundle


def generate_browse_product(session, product):
    # type: (Session, P.Product) -> BrowseProduct
    browse_product = product.browse_product()
    print "completing browse product", browse_product

    make_browse_product(product, browse_product)
    (db_browse_collection,
     db_browse_product) = make_db_browse_product(
        session,
        product,
        browse_product)
    label = make_and_save_product_browse_label(
        db_browse_collection,
        db_browse_product)
    verify_label_or_raise(label)

    # postconditions
    try:
        session.query(BrowseProduct).filter_by(
            lid=str(db_browse_product.lid)).one()
    except:
        assert False, 'browse product %s exists in database' % \
            str(db_browse_product.lid)
    assert os.path.isfile(str(db_browse_product.browse_filepath)), \
        'browse product file %s exists' % \
        str(db_browse_product.browse_filepath)
    assert os.path.isfile(str(db_browse_product.label_filepath)), \
        'label for browse product %s exists' % str(db_browse_product.lid)

    return db_browse_product


def complete_fits_product(session, archive, collection, product):
    # type: (Session, A.Archive, C.Collection, P.Product) -> FitsProduct
    print "completing product", product
    db_fits_product = db_add_product(session, archive,
                                     collection, product)
    if db_fits_product is not None:
        # i.e., FITS parsing didn't fail and it isn't a
        # bad_fits_file
        label = make_and_save_product_observational_label(
            db_fits_product)
        verify_label_or_raise(label)

        # Now make browse products
        if collection.suffix() in RAW_SUFFIXES:
            generate_browse_product(session, product)

        # TODO: generate_spice_kernel_product(session, product)

    # postconditions
    try:
        session.query(FitsProduct).filter_by(
            lid=str(db_fits_product.lid)).one()
    except:
        assert False, 'FITS product %s exists in database' % \
            str(db_fits_product.lid)
    assert os.path.isfile(str(db_fits_product.fits_filepath)), \
        'FITS product file %s exists' % \
        str(db_fits_product.fits_filepath)
    assert os.path.isfile(str(db_fits_product.label_filepath)), \
        'label for FITS product %s exists' % str(db_fits_product.lid)

    return db_fits_product


def complete_doc_collection(session, db_bundle, doc_collection):
    # type: (Session, Bundle, C.Collection) -> DocumentCollection
    db_collection = db_add_document_collection(session,
                                               doc_collection)
    for product in doc_collection.products():
        print "making document_product", product
        db_doc_product = db_add_document_product(session, product)
        label = make_and_save_product_document_label(db_bundle, db_doc_product)
        verify_label_or_raise(label)

        # postconditions
        try:
            session.query(DocumentProduct).filter_by(
                lid=str(db_doc_product.lid)).one()
        except:
            assert False, 'Document product %s exists in database' % \
                str(db_doc_product.lid)
        assert os.path.isfile(str(db_doc_product.fits_filepath)), \
            'Document product file %s exists' % \
            str(db_doc_product.fits_filepath)
        assert os.path.isfile(str(db_doc_product.label_filepath)), \
            'label for document product %s exists' % str(db_doc_product.lid)

    print "making collection label", doc_collection
    # TODO inventory
    make_and_save_product_collection_label(db_collection)

    # postconditions
    try:
        session.query(DocumentCollection).filter_by(
            lid=str(db_collection.lid)).one()
    except:
        assert False, 'Document collection %s exists in database' % \
            str(db_collection.lid)
    assert os.path.isdir(str(db_collection.full_filepath)), \
        'document collection directory  %s exists' % \
        str(db_collection.fits_filepath)
    assert os.path.isfile(str(db_collection.label_filepath)), \
        'label for document collection %s exists' % str(db_collection.lid)
    if False:  # TODO inventory
        assert os.path.isfile(str(db_collection.inventory_filepath)), \
            'inventory for document collection %s exists' % \
            str(db_collection.lid)

    return db_collection


def complete_non_doc_collection(session, archive, bundle, collection):
    # type: (Session, A.Archive, B.Bundle, C.Collection) -> _NDCollection
    print "completing collection", collection
    db_collection = db_add_non_document_collection(session, archive,
                                                   bundle, collection)
    for product in list(collection.products()):
        complete_fits_product(session, archive, collection, product)

    print "making collection label", collection
    # TODO inventory
    label = make_and_save_product_collection_label(db_collection)
    verify_label_or_raise(label)

    # postconditions
    try:
        session.query(NonDocumentCollection).filter_by(
            lid=str(db_collection.lid)).one()
    except:
        assert False, 'Non-document collection %s exists in database' % \
            str(db_collection.lid)
    assert os.path.isdir(str(db_collection.full_filepath)), \
        'Non-document collection directory  %s exists' % \
        str(db_collection.fits_filepath)
    assert os.path.isfile(str(db_collection.label_filepath)), \
        'label for non-document collection %s exists' % str(db_collection.lid)
    if False:  # TODO inventory
        assert os.path.isfile(str(db_collection.inventory_filepath)), \
            'inventory for non-document collection %s exists' % \
            str(db_collection.lid)

    return db_collection


def complete_bundle(session, archive, bundle):
    # type: (Session, A.Archive, B.Bundle) -> Bundle

    # Move FITS info into database and build labels.

    # TODO: Since we might use out-of-order information (say, an
    # observational product might need info from the documents), we
    # need to split the two pieces of functionality: populate the
    # filesystem and database, and only then build labels.
    db_bundle = db_add_bundle(session, archive, bundle)
    print "completing bundle", bundle
    for collection in bundle.collections():
        db_collection = complete_non_doc_collection(session, archive,
                                                    bundle, collection)

    # Move documentation into filesystem and database
    doc_collection = populate_document_collection(bundle)
    if doc_collection:
        print "making document_collection", doc_collection
        db_doc_collection = complete_doc_collection(session,
                                                    db_bundle,
                                                    doc_collection)

    print "making bundle label", bundle
    label = make_and_save_product_bundle_label(db_bundle)
    verify_label_or_raise(label)

    # postconditions
    try:
        session.query(Bundle).filter_by(
            lid=str(db_bundle.lid)).one()
    except:
        assert False, 'Bundle %s exists in database' % \
            str(db_bundle.lid)
    assert os.path.isdir(str(db_bundle.full_filepath)), \
        'Bundle directory  %s exists' % \
        str(db_bundle.fits_filepath)
    assert os.path.isfile(str(db_bundle.label_filepath)), \
        'label for bundle %s exists' % str(db_bundle.lid)

    return db_bundle

BUNDLE_NAME = 'hst_11536'


def reset_bundle(bundle):
    # type: (B.Bundle) -> None
    bundle_filepath = bundle.absolute_filepath()
    print 'bundle_filepath =', bundle_filepath
    for file in os.listdir(bundle_filepath):
        filepath = os.path.join(bundle_filepath, file)
        if file in ['.', '..']:
            pass
        elif file == DATABASE_NAME:
            print 'removing ', filepath
            os.remove(filepath)
        elif file == 'document' or file.startswith('browse_'):
            print 'removing dir ', filepath
            shutil.rmtree(filepath)
        # will also need for SPICE: TODO

    os.system("find %s -name '*.xml' -delete" % bundle_filepath)


def run():
    archive = get_any_archive()
    for bundle in archive.bundles():
        if True or bundle.lid.bundle_id == BUNDLE_NAME:
            print bundle.lid
            reset_bundle(bundle)
            db_filepath = os.path.join(bundle.absolute_filepath(),
                                       DATABASE_NAME)

            session = create_database_tables_and_session(db_filepath)
            complete_bundle(session, archive, bundle)
            session.close()

if __name__ == '__main__':
    run()

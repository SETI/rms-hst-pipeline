import os
import os.path
import shutil

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from pdart.db.SqlAlchLabels import make_browse_product, \
    make_db_browse_product, make_product_browse_label, \
    make_product_bundle_label, make_product_collection_label, \
    make_product_document_label, make_product_observational_label
from pdart.db.SqlAlchTables import Base, Bundle
from pdart.pds4.Archives import get_any_archive
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.xml.Schema import verify_label_or_raise

from SqlAlch import db_add_product, db_add_non_document_collection
from SqlAlchDocs import db_add_document_collection, db_add_document_product, \
    populate_document_collection

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pds4.Archive as A
    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P


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


def complete_bundle(session, archive, bundle):
    # type: (Session, A.Archive, B.Bundle) -> None

    # Move FITS info into database and build labels.

    # TODO: Since we might use out-of-order information (say, an
    # observational product might need info from the documents), we
    # need to split the two pieces of functionality: populate the
    # filesystem and database, and only then build labels.
    db_bundle = db_add_bundle(session, archive, bundle)
    print "completing bundle", bundle
    for collection in bundle.collections():
        print "completing collection", collection
        db_collection = db_add_non_document_collection(session, archive,
                                                       bundle, collection)
        for product in list(collection.products()):
            print "completing product", product
            db_fits_product = db_add_product(session, archive,
                                             collection, product)
            if db_fits_product:
                label = make_product_observational_label(db_fits_product)
                # TODO Does it get put into the filesystem?
                verify_label_or_raise(label)

                # Now make browse products
                if collection.suffix() in RAW_SUFFIXES:
                    browse_product = product.browse_product()
                    print "completing browse product", browse_product

                    make_browse_product(product, browse_product)
                    (db_browse_collection,
                     db_browse_product) = make_db_browse_product(
                        session,
                        product,
                        browse_product)
                    label = make_product_browse_label(db_browse_collection,
                                                      db_browse_product)
                    # TODO Does it get put into the filesystem?
                    verify_label_or_raise(label)

                # TODO Repeat the same for SPICE kernels

        print "making collection label", collection
        label = make_product_collection_label(db_collection)
        # TODO Does it get put into the filesystem?
        verify_label_or_raise(label)

    # Move documentation into filesystem and database
    doc_collection = populate_document_collection(bundle)
    if doc_collection:
        print "making document_collection", doc_collection
        db_doc_collection = db_add_document_collection(session,
                                                       doc_collection)
        for product in doc_collection.products():
            print "making document_product", product
            db_product = db_add_document_product(session, product)
            label = make_product_document_label(db_bundle, db_product)
            verify_label_or_raise(label)

        print "making collection label", doc_collection
        make_product_collection_label(db_doc_collection)

    print "making bundle label", bundle
    label = make_product_bundle_label(db_bundle)
    # Does it get put into the filesystem?  No.
    verify_label_or_raise(label)

BUNDLE_NAME = 'hst_11536'
DATABASE_NAME = 'sqlalch-database.db'


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
        if bundle.lid.bundle_id == BUNDLE_NAME:
            print bundle.lid
            reset_bundle(bundle)
            db_filepath = os.path.join(bundle.absolute_filepath(),
                                       DATABASE_NAME)
            engine = create_engine('sqlite:///' + db_filepath)
            Base.metadata.create_all(engine)
            session = sessionmaker(bind=engine)()
            # the 'bundles' db table doesn't exist.  Why???
            complete_bundle(session, archive, bundle)
            session.close()

if __name__ == '__main__':
    run()

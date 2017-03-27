from sqlalchemy.orm import Session

from pdart.xml.Schema import verify_label_or_raise

from SqlAlch import db_add_product
from SqlAlchDocs import db_add_document_collection, \
    populate_document_collection
from SqlAlchLabels import make_browse_product, make_db_browse_product, \
    make_product_browse_label, make_product_bundle_label, \
    make_product_collection_label

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pds4.Archive as A
    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P


def make_bundle_label(session, bundle):
    # type: (Session, B.Bundle) -> None
    assert False


def make_collection_label(session, collection):
    # type: (Session, C.Collection) -> None
    assert False


def complete_bundle(session, archive, bundle):
    # type: (Session, A.Archive, B.Bundle) -> None

    # Move FITS info into database and build labels.  TODO: Split
    # those two pieces of functionality.
    for collection in bundle.collections():
        for product in collection.products():
            db_add_product(session, archive, collection, product)

    # Now make browse products
    for product in bundle.products():
        browse_product = product.browse_product()
        make_browse_product(product, browse_product)
        (db_browse_collection,
         db_browse_product) = make_db_browse_product(session,
                                                     product,
                                                     browse_product)
        label = make_product_browse_label(db_browse_collection,
                                          db_browse_product)
        # TODO Does it get put into the filesystem?
        verify_label_or_raise(label)

    # TODO Repeat the same for SPICE kernels

    # Label the FITS/browse (/SPICE: TODO) collections.
    for collection in bundle.collections():
        # TODO Wrong kind of Collection
        label = ''  # make_product_collection_label(collection)
        # TODO Does it get put into the filesystem?
        verify_label_or_raise(label)

    # Move documentation into filesystem and database
    doc_collection = populate_document_collection(bundle)
    if doc_collection:
        db_document_collection = db_add_document_collection(session,
                                                            doc_collection)
        # TODO When do we make labels for the products?
        make_product_collection_label(db_document_collection)

    # TODO Wrong kind of Bundle
    label = ''  # make_product_bundle_label(bundle)
    # TODO Does it get put into the filesystem?
    verify_label_or_raise(label)

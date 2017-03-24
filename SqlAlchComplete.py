from SqlAlch import *

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pds4.Archive as A
    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P


def db_add_browse_product(session):
    # type: (Session) -> None
    pass


def generate_browse_product(fits_product):
    # type: (Session) -> None
    pass


def generate_document_collection(session, bundle):
    # type: (Session, B.Bundle) -> None
    pass


def make_bundle_label(session, bundle):
    # type: (Session, B.Bundle) -> None
    pass


def make_collection_label(session, collection):
    # type: (Session, C.Collection) -> None
    pass


def make_product_label(session, product):
    # type: (Session, P.Product) -> None
    pass


def complete_bundle(session, archive, bundle):
    # type: (Session, A.Archive, B.Bundle) -> None

    # Move FITS info into database
    for collection in bundle.collections():
        for product in collection.products():
            db_add_product(session, archive, collection, product)

            # Or do we do this in the database?
    for product in bundle.products():
        generate_browse_product(product)

    for product in bundle.products():
        db_add_browse_product(product)

    for product in bundle.products():
        make_product_label(session, product)

    # Repeat the same for SPICE kernels

    # Move documentation into filesystem and database
    generate_document_collection(session, bundle)

    for collection in bundle.collections():
        make_collection_label(session, collection)

    make_bundle_label(session, bundle)

import os
import os.path

import pdart.add_pds_tools
import picmaker

from pdart.db.SqlAlchTables import BrowseProduct, \
    NonDocumentCollection, Product, db_browse_product_exists, \
    db_non_document_collection_exists

from pdart.pds4.HstFilename import HstFilename

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import AnyStr, Tuple
    from sqlalchemy.orm import Session

    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P

    # a type synonym
    _BrowseCollectionAndProduct = Tuple[NonDocumentCollection, BrowseProduct]


def _ensure_directory(dir):
    # type: (AnyStr) -> None
    """Make the directory if it doesn't already exist."""

    # TODO This is cut-and-pasted from
    # pdart.pds4label.BrowseProductImageReduction.  Refactor and
    # remove.
    try:
        os.mkdir(dir)
    except OSError:
        pass
    assert os.path.isdir(dir), dir


def make_browse_product(fits_product, browse_product):
    # type: (P.Product, P.Product) -> None
    """
    Given FITS product and Browse product objects, create images for
    the browse product and save them to the filesystem.
    """
    # PRECONDITION: the FITS file exists in the filesystem
    filepath = fits_product.first_filepath()
    assert os.path.isfile(filepath)

    basename = os.path.basename(filepath)
    basename = os.path.splitext(basename)[0] + '.jpg'
    browse_collection_dir = browse_product.collection().absolute_filepath()
    _ensure_directory(browse_collection_dir)

    visit = HstFilename(basename).visit()
    target_dir = os.path.join(browse_collection_dir, ('visit_%s' % visit))
    _ensure_directory(target_dir)

    picmaker.ImagesToPics([filepath],
                          target_dir,
                          filter="None",
                          percentiles=(1, 99))
    # POSTCONDITION: browse file exists in the filesystem
    browse_filepath = os.path.join(target_dir, basename)
    assert os.path.isfile(browse_filepath)


def _make_db_browse_collection(session, browse_collection):
    # type: (Session, C.Collection) -> NonDocumentCollection
    lid = str(browse_collection.lid)

    db_browse_collection = \
        session.query(NonDocumentCollection).filter_by(lid=lid).first()

    if not db_browse_collection:
        bundle = browse_collection.bundle()
        db_browse_collection = NonDocumentCollection(
            lid=lid,
            bundle_lid=str(bundle.lid),
            prefix=browse_collection.prefix(),
            suffix=browse_collection.suffix(),
            instrument=browse_collection.instrument(),
            full_filepath=browse_collection.absolute_filepath(),
            label_filepath=browse_collection.label_filepath(),
            inventory_name=browse_collection.inventory_name(),
            inventory_filepath=browse_collection.inventory_filepath())
        session.add(db_browse_collection)
        session.commit()

    # POSTCONDITION
    assert db_non_document_collection_exists(session, browse_collection)
    return db_browse_collection


def make_db_browse_product(session, fits_product, browse_product):
    # type: (Session, P.Product, P.Product) -> _BrowseCollectionAndProduct
    """
    Given a SqlAlchemy session and the FITS and browse product
    objects, create the BrowseCollection and BrowseProduct rows in the
    database.
    """
    # PRECONDITION: the browse product file exists in the filesystem
    assert os.path.isfile(browse_product.first_filepath())

    lid = str(browse_product.lid)

    # TODO I'm deleting any previous record here, but only during
    # development.
    session.query(BrowseProduct).filter_by(product_lid=lid).delete()
    session.query(Product).filter_by(lid=lid).delete()

    browse_filepath = browse_product.absolute_filepath()
    object_length = os.path.getsize(browse_filepath)

    db_browse_product = BrowseProduct(
        lid=str(browse_product.lid),
        collection_lid=str(browse_product.collection().lid),
        label_filepath=browse_product.label_filepath(),
        browse_filepath=browse_filepath,
        object_length=object_length
        )
    session.add(db_browse_product)
    session.commit()

    db_browse_collection = \
        _make_db_browse_collection(session, browse_product.collection())

    # POSTCONDITION
    assert db_browse_product_exists(session, browse_product)
    assert db_non_document_collection_exists(session,
                                             browse_product.collection())

    return (db_browse_collection, db_browse_product)

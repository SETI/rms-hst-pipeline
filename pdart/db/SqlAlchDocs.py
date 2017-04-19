"""
Document information in the database.
"""
import os.path
import shutil
import urllib2
from typing import TYPE_CHECKING

from sqlalchemy import *

from pdart.db.SqlAlchTables import *
from pdart.pds4.Archives import get_any_archive
import pdart.pds4.Collection as C
from pdart.pds4.LID import LID
from pdart.pds4labels.SqlAlchLabels import make_product_document_label
from pdart.xml.Schema import verify_label_or_raise


if TYPE_CHECKING:
    from typing import AnyStr, Tuple

    import pdart.pds4.Bundle as B
    import pdart.pds4.Product as P


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


def _retrieve_doc(url, filepath):
    # type: (unicode, unicode) -> bool
    """Retrieves the text at that URL or raises an exception."""
    try:
        resp = urllib2.urlopen(url)
        contents = resp.read()
        with open(filepath, 'w') as f:
            f.write(contents)
            return True
    except Exception as e:
        return False


def _download_product_documents(proposal_id, product_fp):
    # type: (int, unicode) -> bool
    """
    Using the templates, download the documentation files for this
    proposal ID into the product directory.
    """
    table = [
        ('https://www.stsci.edu/hst/phase2-public/%d.apt', 'phase2.apt'),
        ('https://www.stsci.edu/hst/phase2-public/%d.pdf', 'phase2.pdf'),
        ('https://www.stsci.edu/hst/phase2-public/%d.pro', 'phase2.pro'),
        ('https://www.stsci.edu/hst/phase2-public/%d.prop', 'phase2.prop')
        ]
    # type: List[Tuple[unicode, unicode]]

    downloaded_doc = False

    for (url_template, basename) in table:
        url = url_template % proposal_id
        filepath = os.path.join(product_fp, basename)
        downloaded_doc = _retrieve_doc(url, filepath) or downloaded_doc
    return downloaded_doc


def populate_document_collection(bundle):
    # type: (B.Bundle) -> C.Collection
    """
    Download documentation for the bundle and save in the filesystem.
    """
    bundle_fp = bundle.absolute_filepath()
    collection_fp = os.path.join(bundle_fp, 'document')

    # TODO temporarily erasing and rewriting for development
    try:
        shutil.rmtree(collection_fp)
    except:
        pass
    _ensure_directory(collection_fp)

    product_fp = os.path.join(collection_fp, 'phase2')
    _ensure_directory(product_fp)
    proposal_id = bundle.proposal_id()
    if _download_product_documents(proposal_id, product_fp):
        collection_lid = '%s:document' % str(bundle.lid)
        return C.Collection(bundle.archive, LID(collection_lid))
    else:
        return None


def db_add_document_collection(session, collection):
    # type: (Session, C.Collection) -> Collection
    """
    Given a database session and a Collection object, create a
    DocumentCollection database row, add it to the database, and
    return it.
    """
    db_document_collection = DocumentCollection(
        lid=str(collection.lid),
        bundle_lid=str(collection.bundle().lid),
        full_filepath=collection.absolute_filepath(),
        label_filepath=collection.label_filepath(),
        inventory_name=collection.inventory_name(),
        inventory_filepath=collection.inventory_filepath()
        )
    session.add(db_document_collection)
    session.commit()
    return db_document_collection


def db_add_document_product(session, product):
    # type: (Session, P.Product) -> Product
    """
    Given a database session and a Product object, create a
    DocumentProduct database row, add it to the database, and return
    it.
    """
    db_document_product = DocumentProduct(
        lid=str(product.lid),
        document_filepath=product.absolute_filepath(),
        collection_lid=str(product.collection().lid),
        label_filepath=product.label_filepath())

    session.add(db_document_product)
    for file in product.files():
        db_document_file = DocumentFile(
            product_lid=str(product.lid),
            file_basename=os.path.basename(file.full_filepath())
            )
        session.add(db_document_file)

    session.commit()
    return db_document_product


def _run():
    # type: () -> None
    archive = get_any_archive()
    for bundle in archive.bundles():
        print ('populating %s' % bundle.lid)
        populate_document_collection(bundle)

    DB_FILEPATH = 'trash_me.db'
    try:
        os.remove(DB_FILEPATH)
    except:
        pass
    session = create_database_tables_and_session(DB_FILEPATH)

    for collection in archive.collections():
        if collection.lid.collection_id == 'document':

            bundle = collection.bundle()
            db_bundle = Bundle(lid=str(bundle.lid),
                               proposal_id=bundle.proposal_id(),
                               archive_path=os.path.abspath(archive.root),
                               full_filepath=bundle.absolute_filepath(),
                               label_filepath=bundle.label_filepath())
            session.add(db_bundle)
            session.commit()

            db_add_document_collection(session, collection)
            for product in collection.products():
                db_product = db_add_document_product(session, product)
                label = make_product_document_label(db_bundle, db_product)
                verify_label_or_raise(label)
                print 'verified label for', str(product.lid)

if __name__ == '__main__':
    _run()

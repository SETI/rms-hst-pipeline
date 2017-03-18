import os.path
import shutil
import urllib2
from typing import TYPE_CHECKING

from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

from pdart.pds4.Archives import get_any_archive
from pdart.pds4.LID import LID

from SqlAlchLabels import ensure_directory
from SqlAlchTables import *

if TYPE_CHECKING:
    from typing import Tuple
    from sqlalchemy.orm import Session

    import pdart.pds4.Bundle as B
    import pdart.pds4.Collection as C
    import pdart.pds4.Product as P


def _retrieve_doc(url, filepath):
    # type: (unicode, unicode) -> str
    """Retrieves the text at that URL or raises an exception."""
    try:
        resp = urllib2.urlopen(url)
        contents = resp.read()
        with open(filepath, 'w') as f:
            f.write(contents)
    except Exception as e:
        pass


def _download_product_documents(proposal_id, product_fp):
    # type: (int, unicode) -> None
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
    for (url_template, basename) in table:
        url = url_template % proposal_id
        filepath = os.path.join(product_fp, basename)
        _retrieve_doc(url, filepath)


def populate_document_bundle(bundle):
    # type: (B.Bundle) -> None
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
    ensure_directory(collection_fp)

    product_fp = os.path.join(collection_fp, 'phase2')
    ensure_directory(product_fp)
    proposal_id = bundle.proposal_id()
    _download_product_documents(proposal_id, product_fp)


def db_add_document_collection(session, collection):
    # type: (Session, C.Collection) -> None
    db_document_collection = Collection(
        lid=str(collection.lid),
        bundle_lid=str(collection.bundle().lid),
        prefix='<dummy>',  # TODO
        suffix='<dummy>',  # TODO
        instrument='<dummy>',  # TODO
        full_filepath=collection.absolute_filepath(),
        label_filepath=collection.label_filepath(),
        inventory_name=collection.inventory_name(),
        inventory_filepath=collection.inventory_filepath()
        )
    session.add(db_document_collection)
    session.commit()


def db_add_document_product(session, product):
    # type: (Session, P.Product) -> None
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


def run():
    # type: () -> None
    archive = get_any_archive()
    for bundle in archive.bundles():
        print ('populating %s' % bundle.lid)
        populate_document_bundle(bundle)

    DB_FILEPATH = 'trash_me.db'
    try:
        os.remove(DB_FILEPATH)
    except:
        pass
    engine = create_engine('sqlite:///' + DB_FILEPATH)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    # type: Session

    for collection in archive.collections():
        if collection.lid.collection_id == 'document':
            db_add_document_collection(session, collection)
            for product in collection.products():
                db_add_document_product(session, product)


if __name__ == '__main__':
    run()

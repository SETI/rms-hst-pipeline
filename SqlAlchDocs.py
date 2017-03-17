import os.path
import shutil
import urllib2
from typing import TYPE_CHECKING

from sqlalchemy import *

from pdart.pds4.Archives import get_any_archive
import pdart.pds4.Bundle as B
from pdart.pds4.LID import LID
import pdart.pds4.Product as P

from SqlAlchLabels import ensure_directory
from SqlAlchTables import *

if TYPE_CHECKING:
    from typing import Tuple


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
    eng = create_engine('sqlite:///' + DB_FILEPATH)
    Base.metadata.create_all(eng)

    for collection in archive.collections():
        if collection.lid.collection_id == 'document':
            print ('%s is a document collection' % collection.lid)
            for product in collection.products():
                print ('%s is a document product' % product.lid)


if __name__ == '__main__':
    run()

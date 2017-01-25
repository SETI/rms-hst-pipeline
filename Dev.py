"""
**SCRIPT:** Build the bundle databases then build bundle, collection,
and product labels, and collection inventories.  Uses the bundle
databases.
"""
from contextlib import closing
import glob
import os
import os.path
import sqlite3
import urllib2
import xml.etree.ElementTree

from pdart.db.CreateBundleDatabase import BundleDatabaseCreator
from pdart.db.DatabaseName import DATABASE_NAME
from pdart.db.TableSchemas import *
from pdart.pds4.Archive import Archive
from pdart.pds4.Archives import *
from pdart.pds4.Collection import Collection
from pdart.pds4.LID import LID
from pdart.pds4labels.BrowseProductImageDB import *
from pdart.pds4labels.BrowseProductLabelDB import *
from pdart.pds4labels.BundleLabel import *
from pdart.pds4labels.CollectionLabel import *
from pdart.pds4labels.DBCalls import *
from pdart.pds4labels.ProductLabel import *
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.rules.Combinators import *

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable

VERIFY = False
# type: bool
IN_MEMORY = False
# type: bool
CREATE_DB = False
# type: bool

DOCUMENT = 'document'
# type: unicode


def _get_apt_url(proposal_id):
    # type: (int) -> str
    """Return the URL to get the APT file for the given proposal id."""
    return 'https://www.stsci.edu/hst/phase2-public/%d.apt' % proposal_id


def _get_pdf_url(proposal_id):
    # type: (int) -> str
    """Return the URL to get the PDF file for the given proposal id."""
    return 'https://www.stsci.edu/hst/phase2-public/%d.pdf' % proposal_id


def _get_pro_url(proposal_id):
    # type: (int) -> str
    """Return the URL to get the PRO file for the given proposal id."""
    return 'https://www.stsci.edu/hst/phase2-public/%d.pro' % proposal_id


def _get_prop_url(proposal_id):
    # type: (int) -> str
    """Return the URL to get the PROP file for the given proposal id."""
    return 'https://www.stsci.edu/hst/phase2-public/%d.prop' % proposal_id


def _retrieve_doc(url):
    # type: (unicode) -> str
    """Retrives the text at that URL or raises an exception."""
    resp = urllib2.urlopen(url)
    return resp.read()


def _retrieve_apt(proposal_id, docs_fp):
    # type: (int, unicode) -> None
    """
    Retrieve the APT file for the given proposal id, write it into the
    document directory (creating the directory if necessary), then
    extract the abstract from it and write the abstract into the
    document directory
    """
    apt_xml = _retrieve_doc(_get_apt_url(proposal_id))
    apt_fp = os.path.join(docs_fp, 'phase2.apt')
    with open(apt_fp, 'w') as f:
        f.write(apt_xml)
    print '# Wrote', apt_fp

    abstract_fp = os.path.join(docs_fp, 'abstract.txt')
    root = xml.etree.ElementTree.parse(apt_fp).getroot()
    assert root is not None
    abst = root.find('.//Abstract')
    assert abst is not None
    assert abst.text is not None
    with open(abstract_fp, 'w') as f2:
        f2.write(abst.text)
    print '# Wrote', abstract_fp


def _retrieve_pdf(proposal_id, docs_fp):
    # type: (int, unicode) -> None
    """
    Retrieve the PDF file for the given proposal id and write it
    into the document directory.
    """
    pdf_txt = _retrieve_doc(_get_pdf_url(proposal_id))
    pdf_fp = os.path.join(docs_fp, 'phase2.txt')
    with open(pdf_fp, 'w') as f:
        f.write(pdf_txt)
    print '# Wrote', pdf_fp


def _retrieve_pro(proposal_id, docs_fp):
    # type: (int, unicode) -> None
    """
    Retrieve the PRO file for the given proposal id and write it
    into the document directory.
    """
    pro_txt = _retrieve_doc(_get_pro_url(proposal_id))
    pro_fp = os.path.join(docs_fp, 'phase2.txt')
    with open(pro_fp, 'w') as f:
        f.write(pro_txt)
    print '# Wrote', pro_fp


def _retrieve_prop(proposal_id, docs_fp):
    # type: (int, unicode) -> None
    """
    Retrieve the PROP file for the given proposal id and write it into
    the document directory.
    """
    prop_txt = _retrieve_doc(_get_prop_url(proposal_id))
    prop_fp = os.path.join(docs_fp, 'phase2.txt')
    with open(prop_fp, 'w') as f:
        f.write(prop_txt)
    print '# Wrote', prop_fp


def check_browse_collection(needed, archive, conn, collection_lid):
    # type: (bool, Archive, sqlite3.Connection, unicode) -> None
    lid = LID(collection_lid)
    coll = Collection(archive, lid)

    # TODO: Is this good design or just a hack?  Review.
    if coll.prefix() == 'browse':
        return

    browse_coll = coll.browse_collection()
    browse_coll_exists = os.path.isdir(browse_coll.absolute_filepath())
    if needed:
        assert browse_coll_exists, \
            '%s was needed but not created' % browse_coll

        # Check that for each product with a good FITS file, a browse
        # image product and its label was created.
        with closing(conn.cursor()) as cursor:
            for (prod_id,) in get_all_good_collection_products(cursor,
                                                               collection_lid):
                prod = Product(archive, LID(prod_id))
                browse_prod = prod.browse_product()
                image_prod = browse_prod.absolute_filepath()
                assert os.path.isfile(image_prod), \
                    'browse image %s for %s was not created' % (image_prod,
                                                                prod_id)
                image_label = prod.browse_product().label_filepath()
                assert os.path.isfile(image_label), \
                    'browse label %s for %s was not created' % (image_prod,
                                                                prod_id)
                with closing(conn.cursor()) as cursor2:
                    iter = get_product_info_db(cursor2, browse_prod.lid.lid)
                    if iter:
                        browse_prod_info = list(iter)
                    else:
                        browse_prod_info = []

                assert browse_prod_info, \
                    "Didn't put browse product %s into the database" % \
                    browse_prod

        label_fp = browse_coll.label_filepath()
        assert os.path.isfile(label_fp), \
            'no browse collection label at %s' % label_fp

        inv_fp = browse_coll.inventory_filepath()
        assert os.path.isfile(inv_fp), \
            'no browse inventory at %s' % inv_fp

        with closing(conn.cursor()) as cursor:
            iter2 = get_collection_info_db(cursor, browse_coll.lid.lid)
            if iter2:
                browse_coll_info = list(iter2)
            else:
                browse_coll_info = []

        assert browse_coll_info, \
            "Didn't put browse collection %s into the database" % \
            browse_coll

        # TODO Any more tests?
    else:
        assert not browse_coll_exists, "%s exists but shouldn't" % browse_coll


def needs_browse_collection(collection_lid):
    # type: (unicode) -> bool
    lid = LID(collection_lid)
    prefix = re.match(Collection.DIRECTORY_PATTERN,
                      lid.collection_id).group(1)
    suffix = re.match(Collection.DIRECTORY_PATTERN,
                      lid.collection_id).group(3)
    return prefix == 'data' and suffix in RAW_SUFFIXES


def add_collection(cursor, collection):
    # type: (sqlite3.Cursor, Collection) -> None
    cursor.execute(COLLECTIONS_SQL, collection_tuple(collection))


def make_db_browse_collection_and_label(archive, conn, collection_lid):
    # type: (Archive, sqlite3.Connection, unicode) -> None
    needed = needs_browse_collection(collection_lid)
    if needed:
        # create the products
        make_db_collection_browse_product_images(archive, conn, collection_lid)
        make_db_collection_browse_product_labels(archive, conn, collection_lid)
        collection = Collection(archive, LID(collection_lid))
        browse_collection = collection.browse_collection()
        with closing(conn.cursor()) as cursor:
            add_collection(cursor, browse_collection)
        make_db_collection_label_and_inventory(
            conn, browse_collection.lid.lid, False)
    check_browse_collection(needed, archive, conn, collection_lid)


def check_document_collection(archive, conn, bundle_lid):
    # type: (Archive, sqlite3.Connection, unicode) -> None
    bundle = Bundle(archive, LID(bundle_lid))
    docs_fp = os.path.join(bundle.absolute_filepath(), DOCUMENT)
    assert os.path.isdir(docs_fp), \
        'document collection directory not at %s' % docs_fp

    files = glob.glob(docs_fp + '/*')
    print "files =", files
    assert files, 'no files in ' + docs_fp

    doc_coll = Collection(archive, LID('%s:%s' % (bundle_lid, DOCUMENT)))
    inv_fp = doc_coll.inventory_filepath()

    # TODO Implement this assertion
    assert True or os.path.isfile(inv_fp), \
        'no document collection inventory at %s' % inv_fp
    label_fp = doc_coll.label_filepath()

    # TODO Implement this assertion
    assert True or os.path.isfile(label_fp), \
        'no document collection label at %s' % label_fp

    # TODO Check for products and collection in database


def _get_document(proposal_id, docs_fp):
    # type: (int, unicode) -> None
    # TODO You're trying alternatives here.  Rewrite to use tasks.
    try:
        _retrieve_apt(proposal_id, docs_fp)
        _retrieve_pdf(proposal_id, docs_fp)
        _retrieve_pro(proposal_id, docs_fp)
        # TODO Create the phase2.pdf file
    except urllib2.HTTPError as e:
        print e
        try:
            _retrieve_prop(proposal_id, docs_fp)
        except urllib2.HTTPError as e2:
            print e2


def make_db_document_collection_and_label(archive, conn, bundle_lid):
    # type: (Archive, sqlite3.Connection, unicode) -> None
    print ('**** In progress: Building document collection ' +
           'for %s:document' % bundle_lid)
    bundle = Bundle(archive, LID(bundle_lid))
    docs_fp = os.path.join(bundle.absolute_filepath(), DOCUMENT)
    os.mkdir(docs_fp)

    # TODO Put products, product labels, collection label, inventory
    # into filesystem
    _get_document(bundle.proposal_id(), docs_fp)

    # TODO Insert products and collection into database

    check_document_collection(archive, conn, bundle_lid)


class ArchiveRecursion(object):
    def __init__(self):
        # type: () -> None
        pass

    def run(self, archive):
        """Implements a bottom-up recursion through the archive."""
        # type: (Archive) -> None
        for bundle in archive.bundles():
            database_fp = os.path.join(bundle.absolute_filepath(),
                                       DATABASE_NAME)
            with closing(sqlite3.connect(database_fp)) as conn:
                with closing(conn.cursor()) as collection_cursor:
                    for (coll,) in get_bundle_collections_db(collection_cursor,
                                                             bundle.lid.lid):
                        with closing(conn.cursor()) as product_cursor:
                            prod_iter = get_good_collection_products_db(
                                product_cursor, coll)
                            for (prod,) in prod_iter:
                                self.handle_product(archive, conn, prod)
                        self.handle_collection(archive, conn, coll)
                self.handle_bundle(archive, conn, bundle.lid.lid)

    def handle_bundle(self, archive, conn, bundle_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        pass

    def handle_collection(self, archive, conn, collection_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        pass

    def handle_product(self, archive, conn, product_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        pass


class LabelCreationRecursion(ArchiveRecursion):
    def __init__(self):
        # type: () -> None
        ArchiveRecursion.__init__(self)

    def handle_bundle(self, archive, conn, bundle_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        make_db_bundle_label(conn, bundle_lid, VERIFY)

    def handle_collection(self, archive, conn, collection_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        make_db_collection_label_and_inventory(conn, collection_lid, VERIFY)

    def handle_product(self, archive, conn, product_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        make_db_product_label(conn, product_lid, VERIFY)


class FullCreationRecursion(LabelCreationRecursion):
    def __init__(self):
        # type: () -> None
        LabelCreationRecursion.__init__(self)

    def handle_bundle(self, archive, conn, bundle_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        LabelCreationRecursion.handle_bundle(self, archive,
                                             conn, bundle_lid)

        # Here we build the document collection for the bundle
        make_db_document_collection_and_label(archive, conn, bundle_lid)

    def handle_collection(self, archive, conn, collection_lid):
        # type: (Archive, sqlite3.Connection, unicode) -> None
        LabelCreationRecursion.handle_collection(self, archive,
                                                 conn, collection_lid)

        # Here we build the sibling browse collection to the data
        # collection
        make_db_browse_collection_and_label(archive, conn, collection_lid)


def dev():
    # type: () -> None
    archive = get_any_archive()

    if CREATE_DB:
        BundleDatabaseCreator(archive).create()

    FullCreationRecursion().run(archive)


if __name__ == '__main__':
    raise_verbosely(dev)

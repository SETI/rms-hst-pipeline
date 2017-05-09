import glob
import os
import os.path
import shutil
import sys
import xml.etree.ElementTree as ET

from pdart.browse.SqlAlchBrowse \
    import make_db_browse_product, make_browse_product
from pdart.db.SqlAlchDBName import DATABASE_NAME
from pdart.db.SqlAlchDocs import db_add_document_collection, \
    db_add_document_product, populate_document_collection
from pdart.db.SqlAlchTables import Base, BrowseProduct, Bundle, \
    Collection, create_database_tables_and_session, DocumentCollection, \
    DocumentProduct, FitsProduct, NonDocumentCollection, Product
from pdart.pds4.Archives import get_any_archive
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.pds4labels.SqlAlchLabels \
    import make_and_save_product_browse_label, \
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
    assert_product_is_complete(session, db_browse_product)

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
    assert_product_is_complete(session, db_fits_product)

    return db_fits_product


def _get_apt_files(docs_dir):
    # type: (unicode) -> List[unicode]
    old_dir = os.getcwd()
    try:
        os.chdir(docs_dir)
        return [os.path.join(docs_dir, f) for f in glob.glob('*.apt')]
    except:
        print "ERROR: %s" % sys.exc_info()[0]
        raise
    finally:
        os.chdir(old_dir)


def _extract_apt_info(session, doc_collection):
    # type: (Session, C.Collection) -> None
    """
    If there is an .apt file in the documents collection, extract the
    information within it to the database and to abstract.txt.
    """
    print '#### _extract_apt_info(session, %s)' % doc_collection
    for product in doc_collection.products():
        docs_dir = product.absolute_filepath()
        # this is the directory the set of documentation files lives in
        apt_files = _get_apt_files(docs_dir)
        print '#### looked for apt files in', docs_dir
        if apt_files:
            print '#### found apt:', apt_files
            assert len(apt_files) == 1
            filepath = apt_files[0]
            tree = ET.parse(filepath)
            root = tree.getroot()
            abstract = root.findall('./ProposalInformation/Abstract')[0].text
            abstract_fp = os.path.join(os.path.dirname(filepath),
                                       'abstract.txt')
            with open(abstract_fp, 'w') as f:
                f.write(abstract)
            # TODO Need to add abstract.txt to the database
        else:
            print '#### found no apt files'

        # TODO See downloads.py in the top directory for other
        # needed data.


def complete_doc_collection(session, db_bundle, doc_collection):
    # type: (Session, Bundle, C.Collection) -> DocumentCollection
    _extract_apt_info(session, doc_collection)
    db_collection = db_add_document_collection(session,
                                               doc_collection)
    for product in doc_collection.products():
        print "making document_product", product
        db_doc_product = db_add_document_product(session, product)
        label = make_and_save_product_document_label(db_bundle, db_doc_product)
        verify_label_or_raise(label)

        # postconditions
        assert_product_is_complete(session, db_doc_product)

    print "making collection label", doc_collection
    # TODO inventory
    make_and_save_product_collection_label(db_collection)

    # postconditions
    assert_collection_is_complete(session, db_collection)

    return db_collection


def assert_log(cond, msg):
    # type: (bool, str) -> None
    if not cond:
        print 'ERROR:', msg


def assert_collection_is_complete(session, db_collection):
    # type: (Session, Collection) -> None
    assert_log(session.query(Collection).filter_by(
            lid=str(db_collection.lid)).one() is not None,
               'collection %s exists in database' %
               str(db_collection.lid))
    assert_log(os.path.isdir(str(db_collection.full_filepath)),
               'collection directory %s exists' %
               str(db_collection.full_filepath))
    assert_log(os.path.isfile(str(db_collection.label_filepath)),
               'label for collection %s exists' % str(db_collection.lid))
    if False:  # TODO inventory
        assert_log(os.path.isfile(str(db_collection.inventory_filepath)),
                   'inventory for collection %s exists' %
                   str(db_collection.lid))
    # TODO assert that for each raw collection, there's a browse
    # collection

    print 'Collection %s is complete' % str(db_collection.lid)


def assert_product_is_complete(session, db_product):
    # type: (Session, Product) -> None
    if db_product is None:
        # it was a bad FITS file
        return
    assert_log(session.query(Product).filter_by(
            lid=str(db_product.lid)).one() is not None,
               'product %s exists in database' %
               str(db_product.lid))
    if isinstance(db_product, FitsProduct):
        assert_log(os.path.isfile(str(db_product.fits_filepath)),
                   'product FITS file %s exists' %
                   str(db_product.fits_filepath))
    elif isinstance(db_product, BrowseProduct):
        assert_log(os.path.isfile(str(db_product.browse_filepath)),
                   'product browse file %s exists' %
                   str(db_product.browse_filepath))
    elif isinstance(db_product, DocumentProduct):
        assert_log(os.path.isdir(str(db_product.document_filepath)),
                   'product document directory %s exists' %
                   str(db_product.document_filepath))
        # TODO Check for files inside
    else:
        assert_log(False, 'uncategorized product %s' % str(db_product.lid))

    assert_log(os.path.isfile(str(db_product.label_filepath)),
               'label for product %s exists' % str(db_product.lid))

    print 'Product %s is complete' % str(db_product.lid)


def assert_bundle_is_complete(session, db_bundle):
    # type: (Session, Bundle) -> None
    assert_log(session.query(Bundle).filter_by(
            lid=str(db_bundle.lid)).one() is not None,
               'bundle %s exists in database' %
               str(db_bundle.lid))
    assert_log(os.path.isdir(str(db_bundle.full_filepath)),
               'bundle directory %s exists' %
               str(db_bundle.full_filepath))
    assert_log(os.path.isfile(str(db_bundle.label_filepath)),
               'label for bundle %s exists' % str(db_bundle.lid))
    # TODO assert it has a document collection?

    print 'Bundle %s is complete' % str(db_bundle.lid)


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
    assert_collection_is_complete(session, db_collection)

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
    assert_bundle_is_complete(session, db_bundle)

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
            print 'removing', filepath
            os.remove(filepath)
        elif file == 'document' or file.startswith('browse_'):
            print 'removing directory', filepath
            shutil.rmtree(filepath)
        # will also need for SPICE: TODO

    os.system("find '%s' -name '*.xml' -delete" % bundle_filepath)


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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB
    from pdart.new_db.SqlAlchTables import *


class BundleWalk(object):
    """
    A walk of a bundle, given its BundleDB.

    The visit_xxx() methods will be called twice: once before
    (post=False) visiting its children, and once after (post=True)
    visiting its children.  Override them to have the walk do
    something.

    The _walk_xxx() methods encode the details of walking an object.
    The knowledge of what kind of children an object can have is
    encoded in these methods and only there.  If a new type of
    database object is added (for instance, we don't have SPICE
    products as I write this, but will in the future), we'll change
    only these functions.

    This class exists because it's error-prone to walk the tree by
    hand and to localize necessary changes in code.
    """

    def __init__(self, bundle_db):
        # type: (BundleDB) -> None
        self.db = bundle_db

    def walk(self):
        # type: () -> None
        bundle = self.db.get_bundle()
        self._walk_bundle(bundle)

    ############################################################

    # The structure of a _walk_xxx() method is:
    #    pre-visit xxx
    #    visit xxx's children
    #    post-visit xxx

    def _walk_bundle(self, bundle):
        # type: (Bundle) -> None
        self.visit_bundle(bundle, False)

        for collection in self.db.get_bundle_collections(bundle):
            collection_lidvid = str(collection.lidvid)
            if self.db.document_collection_exists(collection_lidvid):
                self._walk_document_collection(collection)
            elif self.db.non_document_collection_exists(collection_lidvid):
                self._walk_non_document_collection(collection)
            else:
                assert False, ('Missing collection case: %s' %
                               collection_lidvid)

        self.visit_bundle(bundle, True)

    def _walk_document_collection(self, document_collection):
        # type: (DocumentCollection) -> None
        self.visit_document_collection(document_collection, False)

        collection_lidvid = str(document_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            self._walk_document_product(product)

        self.visit_document_collection(document_collection, True)

    def _walk_non_document_collection(self, non_document_collection):
        # type: (NonDocumentCollection) -> None
        self.visit_non_document_collection(non_document_collection, False)

        collection_lidvid = str(non_document_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            product_lidvid = str(product.lidvid)
            if self.db.browse_product_exists(product_lidvid):
                self._walk_browse_product(product)
            elif self.db.fits_product_exists(product_lidvid):
                self._walk_fits_product(product)
            else:
                assert False, ('Missing product case: %s' %
                               product_lidvid)

        self.visit_non_document_collection(non_document_collection, True)

    def _walk_browse_product(self, browse_product):
        # type: (BrowseProduct) -> None
        self.visit_browse_product(browse_product, False)

        product_lidvid = str(browse_product.lidvid)
        browse_file = self.db.get_product_file(product_lidvid)
        self.visit_browse_file(browse_file)

        self.visit_browse_product(browse_product, True)

    def _walk_document_product(self, document_product):
        # type: (DocumentProduct) -> None
        self.visit_document_product(document_product, False)

        product_lidvid = str(document_product.lidvid)
        for document_file in self.db.get_product_files(product_lidvid):
            self.visit_document_file(document_file)
        self.visit_document_file(document_file)

        self.visit_document_product(document_product, True)

    def _walk_fits_product(self, fits_product):
        # type: (FitsProduct) -> None
        self.visit_fits_product(fits_product, False)

        product_lidvid = str(fits_product.lidvid)
        fits_file = self.db.get_product_file(product_lidvid)
        basename = unicode(fits_file.basename)
        if self.db.bad_fits_file_exists(basename, product_lidvid):
            self.visit_bad_fits_file(fits_file)
        elif self.db.fits_file_exists(basename, product_lidvid):
            self.visit_fits_file(fits_file)
        else:
            assert False, ('Missing FITS product case: %s in %s' %
                           (basename, product_lidvid))

        self.visit_fits_product(fits_product, True)

    ############################################################

    def visit_bundle(self, bundle, post):
        # type: (Bundle, bool) -> None
        pass

    ############################################################

    def visit_document_collection(self, document_collection, post):
        # type: (DocumentCollection, bool) -> None
        pass

    def visit_non_document_collection(self, non_document_collection, post):
        # type: (NonDocumentCollection, bool) -> None
        pass

    ############################################################

    def visit_browse_product(self, browse_product, post):
        # type: (BrowseProduct, bool) -> None
        pass

    def visit_document_product(self, document_product, post):
        # type: (DocumentProduct, bool) -> None
        pass

    def visit_fits_product(self, fits_product, post):
        # type: (FitsProduct, bool) -> None
        pass

    ############################################################

    def visit_browse_file(self, BrowseFile):
        # type: (BrowseFile) -> None
        pass

    def visit_document_file(self, DocumentFile):
        # type: (DocumentFile) -> None
        pass

    def visit_fits_file(self, FitsFile):
        # type: (FitsFile) -> None
        pass

    def visit_bad_fits_file(self, BadFitsFile):
        # type: (BadFitsFile) -> None
        pass

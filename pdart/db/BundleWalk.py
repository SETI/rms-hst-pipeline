from typing import cast

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    BadFitsFile,
    BrowseFile,
    BrowseProduct,
    Bundle,
    DocumentCollection,
    DocumentFile,
    DocumentProduct,
    FitsFile,
    FitsProduct,
    NonDocumentCollection,
)


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

    def __init__(self, bundle_db: BundleDB) -> None:
        self.db = bundle_db

    def walk(self) -> None:
        bundle = self.db.get_bundle()
        self._walk_bundle(bundle)

    ############################################################

    # The structure of a _walk_xxx() method is:
    #    pre-visit xxx
    #    visit xxx's children
    #    post-visit xxx

    def _walk_bundle(self, bundle: Bundle) -> None:
        bundle_lidvid = str(bundle.lidvid)
        self.visit_bundle(bundle, False)

        for collection in self.db.get_bundle_collections(bundle_lidvid):
            collection_lidvid = str(collection.lidvid)
            if self.db.document_collection_exists(collection_lidvid):
                self._walk_document_collection(cast(DocumentCollection, collection))
            elif self.db.non_document_collection_exists(collection_lidvid):
                self._walk_non_document_collection(
                    cast(NonDocumentCollection, collection)
                )
            else:
                assert False, "Missing collection case: %s" % collection_lidvid

        self.visit_bundle(bundle, True)

    def _walk_document_collection(
        self, document_collection: DocumentCollection
    ) -> None:
        self.visit_document_collection(document_collection, False)

        collection_lidvid = str(document_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            self._walk_document_product(cast(DocumentProduct, product))

        self.visit_document_collection(document_collection, True)

    def _walk_non_document_collection(
        self, non_document_collection: NonDocumentCollection
    ) -> None:
        self.visit_non_document_collection(non_document_collection, False)

        collection_lidvid = str(non_document_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            product_lidvid = str(product.lidvid)
            if self.db.browse_product_exists(product_lidvid):
                self._walk_browse_product(cast(BrowseProduct, product))
            elif self.db.fits_product_exists(product_lidvid):
                self._walk_fits_product(cast(FitsProduct, product))
            else:
                assert False, "Missing product case: %s" % product_lidvid

        self.visit_non_document_collection(non_document_collection, True)

    def _walk_browse_product(self, browse_product: BrowseProduct) -> None:
        self.visit_browse_product(browse_product, False)

        product_lidvid = str(browse_product.lidvid)
        browse_file = self.db.get_product_file(product_lidvid)
        self.visit_browse_file(cast(BrowseFile, browse_file))

        self.visit_browse_product(browse_product, True)

    def _walk_document_product(self, document_product: DocumentProduct) -> None:
        self.visit_document_product(document_product, False)

        product_lidvid = str(document_product.lidvid)
        for document_file in self.db.get_product_files(product_lidvid):
            self.visit_document_file(cast(DocumentFile, document_file))

        self.visit_document_product(document_product, True)

    def _walk_fits_product(self, fits_product: FitsProduct) -> None:
        self.visit_fits_product(fits_product, False)

        product_lidvid = str(fits_product.lidvid)
        fits_file = self.db.get_product_file(product_lidvid)
        basename = str(fits_file.basename)
        if self.db.bad_fits_file_exists(basename, product_lidvid):
            self.visit_bad_fits_file(cast(BadFitsFile, fits_file))
        elif self.db.fits_file_exists(basename, product_lidvid):
            self.visit_fits_file(cast(FitsFile, fits_file))
        else:
            assert False, "Missing FITS product case: %s in %s" % (
                basename,
                product_lidvid,
            )

        self.visit_fits_product(fits_product, True)

    ############################################################

    def visit_bundle(self, bundle: Bundle, post: bool) -> None:
        pass

    ############################################################

    def visit_document_collection(
        self, document_collection: DocumentCollection, post: bool
    ) -> None:
        pass

    def visit_non_document_collection(
        self, non_document_collection: NonDocumentCollection, post: bool
    ) -> None:
        pass

    ############################################################

    def visit_browse_product(self, browse_product: BrowseProduct, post: bool) -> None:
        pass

    def visit_document_product(
        self, document_product: DocumentProduct, post: bool
    ) -> None:
        pass

    def visit_fits_product(self, fits_product: FitsProduct, post: bool) -> None:
        pass

    ############################################################

    def visit_browse_file(self, browse_file: BrowseFile) -> None:
        pass

    def visit_document_file(self, document_file: DocumentFile) -> None:
        pass

    def visit_fits_file(self, fits_file: FitsFile) -> None:
        pass

    def visit_bad_fits_file(self, bad_fits_file: BadFitsFile) -> None:
        pass

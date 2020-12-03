from typing import cast

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    BadFitsFile,
    BrowseFile,
    BrowseProduct,
    Bundle,
    Collection,
    ContextCollection,
    ContextProduct,
    DocumentCollection,
    DocumentFile,
    DocumentProduct,
    FitsFile,
    FitsProduct,
    SchemaCollection,
    SchemaProduct,
    OtherCollection,
    switch_on_collection_subtype,
)
from pdart.pds4.LIDVID import LIDVID


class BundleWalk(object):
    """
    A walk of a bundle, given its BundleDB.

    The visit_xxx() methods will be called twice: once before
    (post=False) visiting its children, and once after (post=True)
    visiting its children.  Override them to have the walk do
    something.

    The __walk_xxx() methods encode the details of walking an object.
    The knowledge of what kind of children an object can have is
    encoded in these methods and only there.  If a new type of
    database object is added (for instance, we don't have SPICE
    products as I write this, but will in the future), we'll change
    the implementation of only these functions, and only in this
    class.  That is why they are named private, with double
    underscores: they should not be overridden in child classes.

    This class exists because it's error-prone to walk the tree by
    hand and to localize necessary changes in code.
    """

    def __init__(self, bundle_db: BundleDB, bundle_lidvid: str) -> None:
        self.db = bundle_db
        self.bundle_lidvid = bundle_lidvid

    def walk(self) -> None:
        bundle = self.db.get_bundle(self.bundle_lidvid)
        self.__walk_bundle(bundle)

    ############################################################

    # The structure of a _walk_xxx() method is:
    #    pre-visit xxx
    #    visit xxx's children
    #    post-visit xxx

    def __walk_bundle(self, bundle: Bundle) -> None:
        bundle_lidvid = str(bundle.lidvid)
        self.visit_bundle(bundle, False)

        for collection in self.db.get_bundle_collections(bundle_lidvid):
            # We have to jump through some hoops to apply
            # switch_on_collection_type() (and keep mypy happy).

            def walk_context(coll: Collection) -> None:
                self.__walk_context_collection(
                    bundle_lidvid, cast(ContextCollection, coll)
                )

            def walk_doc(coll: Collection) -> None:
                self.__walk_document_collection(
                    bundle_lidvid, cast(DocumentCollection, coll)
                )

            def walk_sch(coll: Collection) -> None:
                self.__walk_schema_collection(
                    bundle_lidvid, cast(SchemaCollection, coll)
                )

            def walk_other(coll: Collection) -> None:
                self.__walk_other_collection(bundle_lidvid, cast(OtherCollection, coll))

            switch_on_collection_subtype(
                collection, walk_context, walk_doc, walk_sch, walk_other
            )(collection)

        self.visit_bundle(bundle, True)

    def __walk_context_collection(
        self, bundle_lidvid: str, context_collection: ContextCollection
    ) -> None:
        self.visit_context_collection(bundle_lidvid, context_collection, False)

        collection_lidvid = str(context_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            self.__walk_context_product(
                collection_lidvid, cast(ContextProduct, product)
            )

        self.visit_context_collection(bundle_lidvid, context_collection, True)

    def __walk_document_collection(
        self, bundle_lidvid: str, document_collection: DocumentCollection
    ) -> None:
        self.visit_document_collection(bundle_lidvid, document_collection, False)

        collection_lidvid = str(document_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            self.__walk_document_product(
                collection_lidvid, cast(DocumentProduct, product)
            )

        self.visit_document_collection(bundle_lidvid, document_collection, True)

    def __walk_schema_collection(
        self, bundle_lidvid: str, schema_collection: SchemaCollection
    ) -> None:
        self.visit_schema_collection(bundle_lidvid, schema_collection, False)

        collection_lidvid = str(schema_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            self.__walk_schema_product(collection_lidvid, cast(SchemaProduct, product))

        self.visit_schema_collection(bundle_lidvid, schema_collection, True)

    def __walk_other_collection(
        self, bundle_lidvid: str, other_collection: OtherCollection
    ) -> None:
        self.visit_other_collection(bundle_lidvid, other_collection, False)

        collection_lidvid = str(other_collection.lidvid)
        for product in self.db.get_collection_products(collection_lidvid):
            product_lidvid = str(product.lidvid)
            if self.db.browse_product_exists(product_lidvid):
                self.__walk_browse_product(
                    collection_lidvid, cast(BrowseProduct, product)
                )
            elif self.db.fits_product_exists(product_lidvid):
                self.__walk_fits_product(collection_lidvid, cast(FitsProduct, product))
            else:
                assert False, f"Missing product case: {product_lidvid}"

        self.visit_other_collection(bundle_lidvid, other_collection, True)

    def __walk_browse_product(
        self, collection_lidvid: str, browse_product: BrowseProduct
    ) -> None:
        self.visit_browse_product(collection_lidvid, browse_product, False)

        product_lidvid = str(browse_product.lidvid)
        browse_file = self.db.get_product_file(product_lidvid)
        self.visit_browse_file(collection_lidvid, cast(BrowseFile, browse_file))

        self.visit_browse_product(collection_lidvid, browse_product, True)

    def __walk_context_product(
        self, collection_lidvid: str, context_product: ContextProduct
    ) -> None:
        self.visit_context_product(collection_lidvid, context_product, False)
        self.visit_context_product(collection_lidvid, context_product, True)

    def __walk_schema_product(
        self, collection_lidvid: str, schema_product: SchemaProduct
    ) -> None:
        self.visit_schema_product(collection_lidvid, schema_product, False)
        self.visit_schema_product(collection_lidvid, schema_product, True)

    def __walk_document_product(
        self, collection_lidvid: str, document_product: DocumentProduct
    ) -> None:
        self.visit_document_product(collection_lidvid, document_product, False)

        product_lidvid = str(document_product.lidvid)
        for document_file in self.db.get_product_files(product_lidvid):
            self.visit_document_file(
                collection_lidvid, cast(DocumentFile, document_file)
            )

        self.visit_document_product(collection_lidvid, document_product, True)

    def __walk_fits_product(
        self, collection_lidvid: str, fits_product: FitsProduct
    ) -> None:
        self.visit_fits_product(collection_lidvid, fits_product, False)

        product_lidvid = str(fits_product.lidvid)
        fits_file = self.db.get_product_file(product_lidvid)
        basename = str(fits_file.basename)
        if self.db.bad_fits_file_exists(basename, product_lidvid):
            self.visit_bad_fits_file(collection_lidvid, cast(BadFitsFile, fits_file))
        elif self.db.fits_file_exists(basename, product_lidvid):
            self.visit_fits_file(collection_lidvid, cast(FitsFile, fits_file))
        else:
            assert False, "Missing FITS product case: {basename} in {product_lidvid}u"

        self.visit_fits_product(collection_lidvid, fits_product, True)

    ############################################################

    def visit_bundle(self, bundle: Bundle, post: bool) -> None:
        pass

    ############################################################

    def visit_context_collection(
        self, bundle_lidvid: str, context_collection: ContextCollection, post: bool
    ) -> None:
        pass

    def visit_document_collection(
        self, bundle_lidvid: str, document_collection: DocumentCollection, post: bool
    ) -> None:
        pass

    def visit_schema_collection(
        self, bundle_lidvid: str, schema_collection: SchemaCollection, post: bool
    ) -> None:
        pass

    def visit_other_collection(
        self, bundle_lidvid: str, other_collection: OtherCollection, post: bool
    ) -> None:
        pass

    ############################################################

    def visit_browse_product(
        self, collection_lidvid: str, browse_product: BrowseProduct, post: bool
    ) -> None:
        pass

    def visit_context_product(
        self, collection_lidvid: str, context_product: ContextProduct, post: bool
    ) -> None:
        pass

    def visit_document_product(
        self, collection_lidvid: str, document_product: DocumentProduct, post: bool
    ) -> None:
        pass

    def visit_schema_product(
        self, collection_lidvid: str, schema_product: SchemaProduct, post: bool
    ) -> None:
        pass

    def visit_fits_product(
        self, collection_lidvid: str, fits_product: FitsProduct, post: bool
    ) -> None:
        pass

    ############################################################

    def visit_browse_file(
        self, collection_lidvid: str, browse_file: BrowseFile
    ) -> None:
        pass

    def visit_document_file(
        self, collection_lidvid: str, document_file: DocumentFile
    ) -> None:
        pass

    def visit_fits_file(self, collection_lidvid: str, fits_file: FitsFile) -> None:
        pass

    def visit_bad_fits_file(
        self, collection_lidvid: str, bad_fits_file: BadFitsFile
    ) -> None:
        pass

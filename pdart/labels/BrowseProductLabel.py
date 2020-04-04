"""
Functionality to create a label for a browse product containing browse
images.
"""
from typing import cast

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import (
    BrowseFile,
    BrowseProduct,
    Collection,
    File,
    NonDocumentCollection,
    Product,
)
from pdart.labels.BrowseProductLabelXml import make_label
from pdart.labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify


def make_browse_product_label(
    bundle_db: BundleDB,
    browse_product_lidvid: str,
    browse_file_basename: str,
    verify: bool,
) -> bytes:
    """
    Create the label text for the browse product having the given
    LIDVID using the bundle database.  If verify is True, verify the
    label against its XML and Schematron schemas.  Raise an exception
    if either fails.
    """
    product: Product = bundle_db.get_product(browse_product_lidvid)
    assert isinstance(product, BrowseProduct)
    browse_product: BrowseProduct = cast(BrowseProduct, product)
    fits_product_lidvid = browse_product.fits_product_lidvid
    file: File = bundle_db.get_file(browse_file_basename, browse_product_lidvid)
    assert isinstance(file, BrowseFile)
    browse_file: BrowseFile = cast(BrowseFile, file)

    browse_collection_lidvid = browse_product.collection_lidvid
    collection: Collection = bundle_db.get_collection(browse_collection_lidvid)
    assert isinstance(collection, NonDocumentCollection)
    browse_collection: NonDocumentCollection = cast(NonDocumentCollection, collection)

    bundle = bundle_db.get_bundle()

    label = (
        make_label(
            {
                "proposal_id": bundle.proposal_id,
                "suffix": browse_collection.suffix,
                "browse_lid": lidvid_to_lid(browse_product_lidvid),
                "browse_vid": lidvid_to_vid(browse_product_lidvid),
                "data_lidvid": fits_product_lidvid,
                "browse_file_name": browse_file_basename,
                "object_length": browse_file.byte_size,
            }
        )
        .toxml()
        .encode()
    )

    return pretty_and_verify(label, verify)

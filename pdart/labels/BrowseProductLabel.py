"""
Functionality to create a label for a browse product containing browse
images.
"""

from pdart.db.bundle_db import bundle_db
from pdart.db.sql_alch_tables import (
    BrowseFile,
    BrowseProduct,
    Collection,
    File,
    OtherCollection,
    Product,
)
from pdart.labels.BrowseProductLabelXml import make_label
from pdart.labels.LabelError import LabelError
from pdart.labels.utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify


def make_browse_product_label(
    bundle_db: bundle_db,
    browse_collection_lidvid: str,
    browse_product_lidvid: str,
    browse_file_basename: str,
    bundle_lidvid: str,
    verify: bool,
) -> bytes:
    """
    Create the label text for the browse product having the given
    LIDVID using the bundle database.  If verify is True, verify the
    label against its XML and Schematron schemas.  Raise an exception
    if either fails.
    """
    product: Product = bundle_db.get_product(browse_product_lidvid)

    if not isinstance(product, BrowseProduct):
        raise TypeError(f"{product} is not a BrowseProduct.")
    browse_product: BrowseProduct = product

    fits_product_lidvid = browse_product.fits_product_lidvid
    file: File = bundle_db.get_file(browse_file_basename, browse_product_lidvid)
    if not isinstance(file, BrowseFile):
        raise TypeError(f"{file} is not a BrowseFile.")
    browse_file: BrowseFile = file

    collection: Collection = bundle_db.get_collection(browse_collection_lidvid)

    if not isinstance(collection, OtherCollection):
        raise TypeError(f"{collection} is not a OtherCollection.")
    browse_collection: OtherCollection = collection

    bundle = bundle_db.get_bundle(bundle_lidvid)

    try:
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
    except Exception as e:
        raise LabelError(browse_product_lidvid, browse_file_basename) from e

    return pretty_and_verify(label, verify)

from typing import TYPE_CHECKING

from pdart.new_labels.BrowseProductLabelXml import make_label
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_browse_product_label(bundle_db,
                              fits_product_lidvid,
                              browse_product_lidvid,
                              file_basename,
                              verify):
    # type: (BundleDB, str, str, unicode, bool) -> unicode
    browse_file = bundle_db.get_file(browse_product_lidvid,
                                     file_basename)

    browse_product = bundle_db.get_product(browse_product_lidvid)

    browse_collection_lidvid = browse_product.collection_lidvid
    browse_collection = bundle_db.get_collection(browse_collection_lidvid)

    bundle = bundle_db.get_bundle()

    # TODO These two functions want to be utility functions elsewhere.
    from pdart.pds4.LIDVID import LIDVID

    def lidvid_to_lid(lidvid):
        # type: (str) -> str
        return str(LIDVID(lidvid).lid())

    def lidvid_to_vid(lidvid):
        # type: (str) -> str
        return str(LIDVID(lidvid).vid())

    label = make_label({
        'proposal_id': bundle.proposal_id,
        'suffix': browse_collection.suffix,
        'browse_lid': lidvid_to_lid(browse_product_lidvid),
        'browse_vid': lidvid_to_vid(browse_product_lidvid),
        'data_lidvid': fits_product_lidvid,
        'browse_file_name': file_basename,
        'object_length': browse_file.byte_size
    }).toxml()

    return pretty_and_verify(label, verify)

from typing import TYPE_CHECKING

from pdart.new_labels.BrowseProductLabelXml import make_label
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_browse_product_label(bundle_db, fits_product_lidvid,
                              browse_product_lidvid,
                              collection_lidvid, file_basename,
                              byte_size, verify):
    # type: (BundleDB, str, str, str, unicode, int, bool) -> unicode

    proposal_id = 13012  # TODO from bundle_db
    suffix = 'raw'  # TODO from collection_lidvid

    # TODO use LIDVIDs instead of LIDs in label
    from pdart.pds4.LIDVID import LIDVID

    def lidvid_to_lid(lidvid):
        # type: (str) -> str
        return str(LIDVID(lidvid).lid())

    label = make_label({
        'proposal_id': proposal_id,
        'suffix': suffix,
        'browse_lid': lidvid_to_lid(browse_product_lidvid),
        'data_lid': lidvid_to_lid(fits_product_lidvid),
        'browse_file_name': file_basename,
        'object_length': byte_size
    }).toxml()

    return pretty_and_verify(label, verify)

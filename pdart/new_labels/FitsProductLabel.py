from typing import TYPE_CHECKING

from pdart.new_labels.FitsProductLabelXml import make_label
from pdart.xml.Pretty import pretty_print
from pdart.xml.Schema import verify_label_or_raise

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def make_fits_product_label(bundle_db, product_lidvid, verify):
    # type: (BundleDB, string, bool) -> unicode
    """
    Create the label text for the product having this LIDVID using the
    bundle database.  If verify is True, verify the label against its
    XML and Schematron schemas.  Raise an exception if either fails.
    """
    (lid, vid) = product_lidvid.split('::')
    label = make_label({
        'lid': lid
    })

    return pretty_and_verify(label)


def pretty_and_verify(label, verify):
    # type: (unicode) -> unicode
    label = pretty_print(label)
    if verify:
        verify_label_or_raise(label)
    return label

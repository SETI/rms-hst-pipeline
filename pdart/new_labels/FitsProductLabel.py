from typing import TYPE_CHECKING

from pdart.new_labels.FileContents import get_file_contents
from pdart.new_labels.FitsProductLabelXml import make_label
from pdart.pds4.LIDVID import LIDVID
from pdart.xml.Pretty import pretty_print
from pdart.xml.Schema import verify_label_or_raise

if TYPE_CHECKING:
    from typing import Any, Dict, List
    from pdart.new_db.BundleDB import BundleDB


def make_fits_product_label(bundle_db, card_dicts, product_lidvid, verify):
    # type: (BundleDB, List[Dict[str, Any]], string, bool) -> unicode
    """
    Create the label text for the product having this LIDVID using the
    bundle database.  If verify is True, verify the label against its
    XML and Schematron schemas.  Raise an exception if either fails.
    """
    lidvid = LIDVID(product_lidvid)

    proposal_id = 13012
    suffix = 'raw'
    file_name = 'jbz504eoq_raw.fits'

    label = make_label({
        'lid': str(lidvid.lid()),
        'proposal_id': str(proposal_id),
        'suffix': suffix,
        'file_name': file_name,
        'file_contents': get_file_contents(bundle_db.session,
                                           card_dicts, product_lidvid),
        'Investigation_Area_name': mk_Investigation_Area_name(proposal_id),
        'investigation_lidvid': mk_Investigation_Area_lidvid(proposal_id),
        'Observing_System': observing_system(instrument),
        'Time_Coordinates': time_coordinates(product_lidvid, card_dicts),
        'Target_Identification': get_target(card_dicts),
        'HST': hst_parameters(card_dicts, instrument, product_lidvid)
    })

    return pretty_and_verify(label)


def pretty_and_verify(label, verify):
    # type: (unicode) -> unicode
    label = pretty_print(label)
    if verify:
        verify_label_or_raise(label)
    return label

from typing import TYPE_CHECKING

from pdart.new_labels.FileContents import get_file_contents
from pdart.new_labels.FitsProductLabelXml import make_label, \
    mk_Investigation_Area_lidvid, mk_Investigation_Area_name
from pdart.new_labels.HstParameters import get_hst_parameters
from pdart.new_labels.ObservingSystem import observing_system
from pdart.new_labels.TargetIdentification import get_target
from pdart.new_labels.TimeCoordinates import get_time_coordinates
from pdart.pds4.LIDVID import LIDVID
from pdart.xml.Pretty import pretty_and_verify

if TYPE_CHECKING:
    from typing import Any, Dict, List
    from pdart.new_db.BundleDB import BundleDB


def make_fits_product_label(bundle_db, card_dicts, product_lidvid,
                            file_basename, verify):
    # type: (BundleDB, List[Dict[str, Any]], string, unicode, bool) -> unicode
    """
    Create the label text for the product having this LIDVID using the
    bundle database.  If verify is True, verify the label against its
    XML and Schematron schemas.  Raise an exception if either fails.
    """
    lidvid = LIDVID(product_lidvid)

    product = bundle_db.get_product(product_lidvid)
    collection_lidvid = product.collection_lidvid

    collection = bundle_db.get_collection(collection_lidvid)
    instrument = collection.instrument
    suffix = collection.suffix
    bundle_lidvid = collection.bundle_lidvid

    bundle = bundle_db.get_bundle(bundle_lidvid)
    proposal_id = bundle.proposal_id

    label = make_label({
        'lid': str(lidvid.lid()),
        'proposal_id': str(proposal_id),
        'suffix': suffix,
        'file_name': file_basename,
        'file_contents': get_file_contents(bundle_db.session,
                                           card_dicts, product_lidvid),
        'Investigation_Area_name': mk_Investigation_Area_name(proposal_id),
        'investigation_lidvid': mk_Investigation_Area_lidvid(proposal_id),
        'Observing_System': observing_system(instrument),
        'Time_Coordinates': get_time_coordinates(product_lidvid, card_dicts),
        'Target_Identification': get_target(card_dicts),
        'HST': get_hst_parameters(card_dicts, instrument, product_lidvid)
    }).toxml()

    return pretty_and_verify(label, verify)
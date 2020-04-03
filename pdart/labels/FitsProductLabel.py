"""
Functionality to create a label for a data product containing a single
FITS file.
"""
from pdart.labels.FileContents import get_file_contents
from pdart.labels.FitsProductLabelXml import (
    make_label,
    mk_Investigation_Area_lidvid,
    mk_Investigation_Area_name,
)
from pdart.labels.HstParameters import get_hst_parameters
from pdart.labels.ObservingSystem import observing_system
from pdart.labels.TargetIdentification import get_target
from pdart.labels.TimeCoordinates import get_time_coordinates
from pdart.labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import NonDocumentCollection


def make_fits_product_label(
    bundle_db: BundleDB, product_lidvid: str, file_basename: str, verify: bool
) -> bytes:
    """
    Create the label text for the product having this LIDVID using the
    bundle database.  If verify is True, verify the label against its
    XML and Schematron schemas.  Raise an exception if either fails.
    """
    card_dicts = bundle_db.get_card_dictionaries(product_lidvid, file_basename)
    shm_card_dicts = bundle_db.get_shm_card_dictionaries(product_lidvid, file_basename)
    product = bundle_db.get_product(product_lidvid)
    collection_lidvid = product.collection_lidvid

    collection = bundle_db.get_collection(collection_lidvid)
    assert isinstance(collection, NonDocumentCollection)
    instrument = collection.instrument
    suffix = collection.suffix
    bundle_lidvid = collection.bundle_lidvid

    bundle = bundle_db.get_bundle()
    assert bundle.lidvid == bundle_lidvid
    proposal_id = bundle.proposal_id

    label = (
        make_label(
            {
                "lid": lidvid_to_lid(product_lidvid),
                "vid": lidvid_to_vid(product_lidvid),
                "proposal_id": str(proposal_id),
                "suffix": suffix,
                "file_name": file_basename,
                "file_contents": get_file_contents(
                    bundle_db, card_dicts, instrument, product_lidvid
                ),
                "Investigation_Area_name": mk_Investigation_Area_name(proposal_id),
                "investigation_lidvid": mk_Investigation_Area_lidvid(proposal_id),
                "Observing_System": observing_system(instrument),
                "Time_Coordinates": get_time_coordinates(product_lidvid, card_dicts),
                "Target_Identification": get_target(card_dicts),
                "HST": get_hst_parameters(
                    card_dicts, shm_card_dicts, instrument, product_lidvid
                ),
            }
        )
        .toxml()
        .encode()
    )

    return pretty_and_verify(label, verify)

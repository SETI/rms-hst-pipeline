"""
Functionality to create a label for a data product containing a single
FITS file.
"""
from typing import Any, Dict, List
import os.path

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import OtherCollection
from pdart.labels.FileContents import get_file_contents
from pdart.labels.FitsProductLabelXml import (
    make_label,
    mk_Investigation_Area_lidvid,
    mk_Investigation_Area_name,
)
from pdart.labels.HstParameters import get_hst_parameters
from pdart.labels.LabelError import LabelError
from pdart.labels.ObservingSystem import (
    instrument_host_lid,
    observing_system,
    observing_system_lid,
)
from pdart.labels.TargetIdentification import get_target, get_target_info
from pdart.labels.TimeCoordinates import get_start_stop_times, get_time_coordinates
from pdart.labels.Utils import lidvid_to_lid, lidvid_to_vid
from pdart.xml.Pretty import pretty_and_verify

_MISSING: str = "MISSING_KEYS.txt"


def _look_for_values(
    working_dir: str,
    product_lidvid: str,
    file_basename: str,
    missing_key: str,
    raw_card_dicts: List[Dict[str, Any]],
    shm_card_dicts: List[Dict[str, Any]],
) -> None:
    try:
        raw_value = raw_card_dicts[0][missing_key]
    except:
        raw_value = None

    try:
        shm_value = shm_card_dicts[0][missing_key]
    except:
        shm_value = None

    if raw_value or shm_value:
        msg = (
            f"{product_lidvid}/{file_basename} missing {missing_key}:"
            f"raw={raw_value!r}; shm={shm_value!r}\n"
        )
    else:
        msg = (
            f"{product_lidvid}/{file_basename} missing {missing_key}: "
            "no value is raw or shm\n"
        )
    with open(os.path.join(working_dir, _MISSING), "a") as f:
        f.write(msg)


def make_fits_product_label(
    working_dir: str,
    bundle_db: BundleDB,
    product_lidvid: str,
    file_basename: str,
    verify: bool,
) -> bytes:
    """
    Create the label text for the product having this LIDVID using the
    bundle database.  If verify is True, verify the label against its
    XML and Schematron schemas.  Raise an exception if either fails.
    """
    card_dicts = bundle_db.get_card_dictionaries(product_lidvid, file_basename)
    raw_card_dicts = bundle_db.get_raw_card_dictionaries(product_lidvid, file_basename)
    shm_card_dicts = bundle_db.get_shm_card_dictionaries(product_lidvid, file_basename)
    product = bundle_db.get_product(product_lidvid)
    collection_lidvid = product.collection_lidvid

    collection = bundle_db.get_collection(collection_lidvid)
    assert isinstance(collection, OtherCollection)
    instrument = collection.instrument
    suffix = collection.suffix
    bundle_lidvid = collection.bundle_lidvid

    bundle = bundle_db.get_bundle()
    assert bundle.lidvid == bundle_lidvid
    proposal_id = bundle.proposal_id

    investigation_area_lidvid = mk_Investigation_Area_lidvid(proposal_id)
    bundle_db.create_context_product(investigation_area_lidvid)
    bundle_db.create_context_product(instrument_host_lid())
    bundle_db.create_context_product(observing_system_lid(instrument))
    target_info = get_target_info(card_dicts)
    bundle_db.create_context_product(target_info["lid"])

    # expanding to check SHM files doesn't help
    start_stop_times = get_start_stop_times(card_dicts, raw_card_dicts)

    try:
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
                    "investigation_lidvid": investigation_area_lidvid,
                    "Observing_System": observing_system(instrument),
                    "Time_Coordinates": get_time_coordinates(start_stop_times),
                    "Target_Identification": get_target(target_info),
                    "HST": get_hst_parameters(
                        card_dicts, raw_card_dicts, shm_card_dicts, instrument
                    ),
                }
            )
            .toxml()
            .encode()
        )
    #    except KeyError as e:
    #        missing_key: str = e.args[0]
    #        _look_for_values(
    #            working_dir,
    #            product_lidvid,
    #            file_basename,
    #            missing_key,
    #            raw_card_dicts,
    #            shm_card_dicts,
    #        )
    #        raise LabelError(str(e), product_lidvid, file_basename)
    except Exception as e:
        raise LabelError(str(e), product_lidvid, file_basename)

    return pretty_and_verify(label, verify)

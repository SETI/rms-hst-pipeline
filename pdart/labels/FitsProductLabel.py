"""
Functionality to create a label for a data product containing a single
FITS file.
"""
from typing import Any, Dict, List, Optional, Tuple, cast
import os.path

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import File, OtherCollection
from pdart.labels.FileContents import get_file_contents
from pdart.labels.Lookup import Lookup, MultiDictLookup
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
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.xml.Pretty import pretty_and_verify

_KEY_ERROR_DUMP: str = "KEY_ERROR_DUMP.txt"


# TODO: Note this only works in the original case where all VIDs =
# "1.0".
def to_asn_lidvid(product_lidvid: str) -> str:
    lidvid = LIDVID(product_lidvid)
    raw_asn_lid = lidvid.lid().to_other_suffixed_lid("asn")
    asn_lid_as_list: List[str] = list(str(raw_asn_lid))
    # force the last digit of the product part to '0'
    asn_lid_as_list[-1] = "0"
    asn_lid = LID("".join(asn_lid_as_list))
    return str(LIDVID.create_from_lid_and_vid(asn_lid, lidvid.vid()))


def _directory_siblings(
    working_dir: str, bundle_db: BundleDB, product_lidvid: str
) -> List[str]:
    # Look in the mastDownload directory and search for the file with
    # the product_lidvid's basename.  Then return all its siblings'
    # basenames.
    for dirpath, dirnames, filenames in os.walk(
        os.path.join(working_dir, "mastDownload")
    ):
        basename = bundle_db.get_product_file(product_lidvid).basename
        if basename in filenames:
            return sorted(filenames)
    return []


def _sibling_file(siblings: List[str], suffix: str) -> Optional[str]:
    # Given a list of siblings, return the first one that ends with
    # "_<suffix>.fits".
    ending = f"_{suffix.lower()}.fits"
    for basename in siblings:
        if basename.lower().endswith(ending):
            return basename
    return None


def _association_card_dicts_and_lookup(
    bundle_db: BundleDB,
    product_lidvid: str,
    file_basename: str,
    asn_product_lidvid: str,
) -> Tuple[List[Dict[str, Any]], Lookup]:

    # First check in the actual file, then check *any* of the
    # associated files.  TODO That is probably overkill.  Also, some
    # values need to be combined (like date-time).  Fix it.
    card_dicts = bundle_db.get_card_dictionaries(product_lidvid, file_basename)

    dicts: List[Tuple[str, List[Dict[str, Any]]]] = [(product_lidvid, card_dicts)]

    associated_dicts = [
        (
            prod.lidvid,
            bundle_db.get_card_dictionaries(
                prod.lidvid, bundle_db.get_product_file(prod.lidvid).basename
            ),
        )
        for prod in bundle_db.get_associated_key_products(asn_product_lidvid)
    ]
    dicts.extend(associated_dicts)

    lookup = MultiDictLookup(dicts)
    return card_dicts, lookup


def _triple_card_dicts_and_lookup(
    bundle_db: BundleDB, product_lidvid: str, file_basename: str
) -> Tuple[List[Dict[str, Any]], Lookup]:
    card_dicts = bundle_db.get_card_dictionaries(product_lidvid, file_basename)
    lookup = MultiDictLookup(
        [
            (product_lidvid, card_dicts),
            bundle_db.get_other_suffixed_card_dictionaries_and_lidvid(
                product_lidvid, file_basename, "raw"
            ),
            bundle_db.get_other_suffixed_card_dictionaries_and_lidvid(
                product_lidvid, file_basename, "d0f"
            ),
            bundle_db.get_other_suffixed_card_dictionaries_and_lidvid(
                product_lidvid, file_basename, "shm"
            ),
        ]
    )
    return card_dicts, lookup


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

    product = bundle_db.get_product(product_lidvid)
    collection_lidvid = product.collection_lidvid

    collection = bundle_db.get_collection(collection_lidvid)
    assert isinstance(collection, OtherCollection)
    instrument = collection.instrument
    suffix = collection.suffix
    bundle_lidvid = collection.bundle_lidvid

    siblings = _directory_siblings(working_dir, bundle_db, product_lidvid)
    asn_sib = _sibling_file(siblings, "asn")

    if asn_sib is not None:
        asn_product_lidvid = to_asn_lidvid(product_lidvid)
        assert (
            list(cast(str, LIDVID(asn_product_lidvid).lid().product_id))[-1] == "0"
        ), asn_product_lidvid
        card_dicts, lookup = _association_card_dicts_and_lookup(
            bundle_db, product_lidvid, file_basename, asn_product_lidvid
        )
    else:
        # TODO isn't this quadruple now?
        card_dicts, lookup = _triple_card_dicts_and_lookup(
            bundle_db, product_lidvid, file_basename
        )

    bundle = bundle_db.get_bundle()
    assert bundle.lidvid == bundle_lidvid
    proposal_id = bundle.proposal_id

    investigation_area_lidvid = mk_Investigation_Area_lidvid(proposal_id)
    bundle_db.create_context_product(investigation_area_lidvid)
    bundle_db.create_context_product(instrument_host_lid())
    bundle_db.create_context_product(observing_system_lid(instrument))
    target_info = get_target_info(lookup)
    bundle_db.create_context_product(target_info["lid"])

    try:
        start_stop_times = get_start_stop_times(lookup)
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
                    "HST": get_hst_parameters(lookup, instrument, start_stop_times),
                }
            )
            .toxml()
            .encode()
        )
    except KeyError as e:
        key = e.args[0]

        with open(os.path.join(working_dir, _KEY_ERROR_DUMP), "w") as dump_file:
            print(f"**** LIDVID {product_lidvid}:", file=dump_file)
            print("**** LOOKUP:", lookup, file=dump_file)
            print(
                "**** SIBLINGS:", ", ".join(siblings), file=dump_file,
            )
            asn_sib = _sibling_file(siblings, "asn")
            d0f_sib = _sibling_file(siblings, "d0f")
            raw_sib = _sibling_file(siblings, "raw")
            print("**** ASN FILE:", asn_sib, file=dump_file)
            print("**** D0F FILE:", d0f_sib, file=dump_file)
            print("**** RAW FILE:", raw_sib, file=dump_file)
            if asn_sib is None and d0f_sib is None and raw_sib is None:
                print("**** AY CARAMBA: none of asn, d0f, or raw", file=dump_file)

            if asn_sib is not None:
                assoc_key_prods = bundle_db.get_associated_key_products(
                    asn_product_lidvid
                )
                if assoc_key_prods:
                    print("**** ASSOCIATED KEY PRODUCTS: ****", file=dump_file)
                    for prod in assoc_key_prods:
                        print(prod, file=dump_file)
                else:
                    print(
                        f"**** NO ASSOCIATED KEY PRODUCTS FOR {asn_product_lidvid} ****",
                        file=dump_file,
                    )

            if key == "DATE-OBS":
                lookup.dump_keys(["DATE-OBS", "TIME-OBS", "EXPTIME"], dump_file)
            else:
                lookup.dump_key(key, dump_file)

        raise LabelError(str(e), product_lidvid, file_basename)
    except Exception as e:
        raise LabelError(str(e), product_lidvid, file_basename)

    return pretty_and_verify(label, verify)

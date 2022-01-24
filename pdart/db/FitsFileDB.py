from typing import Any, Dict, List, Optional, Tuple

import astropy.io.fits
import astropy.io.fits.card
from fs.path import basename

from pdart.db.BundleDB import BundleDB
from pdart.db.SqlAlchTables import Association, Card, Hdu
from pdart.pds4.HstFilename import HstFilename

_PYFITS_CARD = Any
_PYFITS_HDU = Any
_PYFITS_OBJ = Any


def _populate_associations(
    db: BundleDB, fits_product_lidvid: str, pyfits_obj: _PYFITS_OBJ
) -> None:
    # Here we blindly assert that the second HDU is a binary
    # table.
    ASSOC_HDU_INDEX = 1
    bin_table = pyfits_obj[ASSOC_HDU_INDEX]  # type: ignore
    if not isinstance(bin_table, astropy.io.fits.BinTableHDU):
        raise TypeError(f"HDU is {type(bin_table)}, not a binary table")

    # Asserting for now that the three columns are named MEMNAME,
    # MEMTYPE and MEMPRSNT.
    column_names = [col.name for col in bin_table.columns]
    if ["MEMNAME", "MEMTYPE", "MEMPRSNT"] != column_names:
        raise ValueError(f"column_names: {column_names} not equal to "
                         + "['MEMNAME', 'MEMTYPE', 'MEMPRSNT']."
                        )

    def create_assoc_dict(
        assoc_indx: int, memname: str, memtype: str, memprsnt: bool
    ) -> Dict[str, Any]:
        return {
            "product_lidvid": fits_product_lidvid,
            "association_index": assoc_indx,
            "hdu_index": ASSOC_HDU_INDEX,
            "memname": memname,
            "memtype": memtype,
            "memprsnt": memprsnt,
        }

    assoc_dicts: List[Dict[str, Any]] = [
        create_assoc_dict(assoc_indx, memname, memtype, memprsnt)
        for (assoc_indx, (memname, memtype, memprsnt)) in enumerate(bin_table.data)
    ]

    db.session.bulk_insert_mappings(Association, assoc_dicts)


def populate_database_from_fits_file(
    db: BundleDB, os_filepath: str, fits_product_lidvid: str
) -> None:
    file_basename = basename(os_filepath)
    try:
        fits = astropy.io.fits.open(os_filepath)

        try:
            db.create_fits_file(
                os_filepath, file_basename, fits_product_lidvid, len(fits)
            )
            _populate_hdus_associations_and_cards(
                db, fits, file_basename, fits_product_lidvid
            )
        finally:
            fits.close()

    except OSError as e:
        db.create_bad_fits_file(os_filepath, file_basename, fits_product_lidvid, str(e))


def _populate_hdus_associations_and_cards(
    db: BundleDB, pyfits_obj: _PYFITS_OBJ, file_basename: str, fits_product_lidvid: str
) -> None:
    def create_hdu_dict(index: int, hdu: _PYFITS_HDU) -> Dict[str, Any]:
        fileinfo = hdu.fileinfo()
        return {
            "product_lidvid": fits_product_lidvid,
            "hdu_index": index,
            "hdr_loc": fileinfo["hdrLoc"],
            "dat_loc": fileinfo["datLoc"],
            "dat_span": fileinfo["datSpan"],
        }

    hdu_dicts = [create_hdu_dict(index, hdu) for index, hdu in enumerate(pyfits_obj)]
    db.session.bulk_insert_mappings(Hdu, hdu_dicts)

    def handle_undefined(val: bool) -> Optional[bool]:
        """Convert undefined values to None"""
        if isinstance(val, astropy.io.fits.card.Undefined):
            return None
        else:
            return val

    def desired_keyword(kw: str) -> bool:
        """Return True if the keyword is wanted"""
        # For now we want all of them.
        return kw is not None

    def create_card_dict(
        hdu_index: int, card_index: int, card: _PYFITS_CARD
    ) -> Dict[str, Any]:
        return {
            "product_lidvid": fits_product_lidvid,
            "hdu_index": hdu_index,
            "card_index": card_index,
            "keyword": card.keyword,
            "value": handle_undefined(card.value),
        }

    card_dicts = [
        create_card_dict(hdu_index, card_index, card)
        for hdu_index, hdu in enumerate(pyfits_obj)
        for card_index, card in enumerate(hdu.header.cards)
    ]
    db.session.bulk_insert_mappings(Card, card_dicts)

    if HstFilename(file_basename).suffix() == "asn":
        _populate_associations(db, fits_product_lidvid, pyfits_obj)

    db.session.commit()


def get_card_dictionaries(
    bundle_db: BundleDB, fits_product_lidvid: str, file_basename: str
) -> List[Dict[str, Any]]:
    return bundle_db.get_card_dictionaries(fits_product_lidvid, file_basename)


def get_file_offsets(
    bundle_db: BundleDB, fits_product_lidvid: str
) -> List[Tuple[int, int, int, int]]:
    return bundle_db.get_file_offsets(fits_product_lidvid)

import pyfits
from fs.path import basename
from typing import TYPE_CHECKING

from pdart.new_db.SqlAlchTables import Card, Hdu

if TYPE_CHECKING:
    from typing import Any, Dict, Tuple
    from pdart.new_db.BundleDB import BundleDB

    # unfortunately, untyped
    _PYFITS_OBJ = Any
    _PYFITS_HDU = Any
    _PYFITS_CARD = Any


def populate_database_from_fits_file(db, os_filepath, fits_product_lidvid):
    # type: (BundleDB, unicode, str) -> None
    file_basename = basename(os_filepath)
    try:
        fits = pyfits.open(os_filepath)

        try:
            db.create_fits_file(file_basename, fits_product_lidvid, len(fits))
            _populate_hdus_and_cards(db,
                                     fits,
                                     file_basename,
                                     fits_product_lidvid)
        finally:
            fits.close()

    except IOError as e:
        db.create_bad_fits_file(file_basename, fits_product_lidvid, e.message)


def _populate_hdus_and_cards(db,
                             pyfits_obj,
                             file_basename,
                             fits_product_lidvid):
    # type: (BundleDB, _PYFITS_OBJ, unicode, str) -> None

    def create_hdu_dict(index, hdu):
        # type: (int, _PYFITS_HDU) -> Dict
        fileinfo = hdu.fileinfo()
        return {'product_lidvid': fits_product_lidvid,
                'hdu_index': index,
                'hdr_loc': fileinfo['hdrLoc'],
                'dat_loc': fileinfo['datLoc'],
                'dat_span': fileinfo['datSpan']}

    hdu_dicts = [create_hdu_dict(index, hdu)
                 for index, hdu in enumerate(pyfits_obj)]
    db.session.bulk_insert_mappings(Hdu, hdu_dicts)

    def handle_undefined(val):
        """Convert undefined values to None"""
        if isinstance(val, pyfits.card.Undefined):
            return None
        else:
            return val

    def desired_keyword(kw):
        # type: (str) -> bool
        """Return True if the keyword is wanted"""
        # For now we want all of them.
        return kw is not None

    def create_card_dict(hdu_index, card_index, card):
        # type: (int, int, _PYFITS_CARD) -> Dict
        return {
            'product_lidvid': fits_product_lidvid,
            'hdu_index': hdu_index,
            'card_index': card_index,
            'keyword': card.keyword,
            'value': handle_undefined(card.value)
        }

    card_dicts = [create_card_dict(hdu_index, card_index, card)
                  for hdu_index, hdu in enumerate(pyfits_obj)
                  for card_index, card in enumerate(hdu.header.cards)]
    db.session.bulk_insert_mappings(Card, card_dicts)
    db.session.commit()


def get_card_dictionaries(bundle_db, fits_product_lidvid, file_basename):
    # type: (BundleDB, str, unicode) -> List[Dict[str, Any]]
    return bundle_db.get_card_dictionaries(fits_product_lidvid,
                                           file_basename)


def get_file_offsets(bundle_db, fits_product_lidvid):
    # type: (BundleDB, unicode) -> List[Tuple[int, int, int, int]]
    return bundle_db.get_file_offsets(fits_product_lidvid)

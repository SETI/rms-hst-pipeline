import pyfits
from fs.path import basename
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def populate_from_fits_file(db, os_filepath, fits_product_lidvid):
    # type: (BundleDB, str, str) -> None
    file_basename = basename(os_filepath)
    try:
        fits = pyfits.open(os_filepath)
        db.create_fits_file(file_basename, fits_product_lidvid)
    except IOError as e:
        print '>>>>', e.message, '!!!!'
        db.create_bad_fits_file(file_basename, fits_product_lidvid, e.message)

import pdart.add_pds_tools
import picmaker  # need to precede this with 'import pdart.add_pds_tools'

from fs.path import basename, join
from typing import TYPE_CHECKING

from pdart.fs.DirUtils import lid_to_dir
from pdart.pds4.LID import LID

if TYPE_CHECKING:
    from fs.base import FS
    from pdart.new_db.BundleDB import BundleDB


# TODO Think this over.  Creation of browse products and population of
# the database need to happen at different times.  Creation of browse
# products happens on a single-version filesystem.  You can't populate
# the database until you've got a multi-version view, becaue you need
# LIDVIDs, not just LIDs.


def populate_database_from_browse_product(db, os_filepath, byte_size,
                                          browse_product_lidvid,
                                          collection_lidvid):
    # type: (BundleDB, unicode, int, str, str) -> None
    file_basename = basename(os_filepath)
    db.create_browse_product(browse_product_lidvid, collection_lidvid)
    db.create_browse_file(file_basename, browse_product_lidvid, byte_size)


def create_browse_directory(fs, browse_product_lid):
    # type: (FS, str) -> None
    browse_dir = lid_to_dir(LID(browse_product_lid))
    fs.makedirs(browse_dir, None, True)


def create_browse_file_from_fits_file(fs, fits_product_lid,
                                      browse_product_lid):
    # type: (FS, str, str) -> None
    fits_basename = LID(fits_product_lid).product_id + '.fits'
    fits_filepath = join(lid_to_dir(LID(fits_product_lid)),
                         fits_basename)
    browse_product_dir = lid_to_dir(LID(browse_product_lid))
    picmaker.ImagesToPics([fs.getsyspath(fits_filepath)],
                          fs.getsyspath(browse_product_dir),
                          filter="None",
                          percentiles=(1, 99))


def _create_browse_product_lid(fits_product_lid):
    # type: (str) -> str
    return str(LID(fits_product_lid).to_browse_lid())

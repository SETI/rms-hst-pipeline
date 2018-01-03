from typing import TYPE_CHECKING

from pdart.new_db.BrowseFileDB import populate_database_from_browse_file

if TYPE_CHECKING:
    from fs.base import FS
    from pdart.new_db.BundleDB import BundleDB


def create_browse_file_from_fits_file_and_populate_database(
        fs, db, fits_product_lid, collection_lidvid):
    # type: (FS, BundleDB, str, str) -> int
    create_browse_file_from_fits_file(fs, fits_product_lid)
    browse_product_lidvid = None  # type: str
    basename = None  # type: unicode
    byte_size = None  # type: int
    populate_database_from_browse_file(
        db, browse_product_lidvid, collection_lidvid, basename, byte_size)
    return 0  # TODO fix this


def create_browse_file_from_fits_file(fs, fits_product_lid):
    # type: (FS, str) -> bool
    return False

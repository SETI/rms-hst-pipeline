from typing import TYPE_CHECKING

from pdart.new_db.BrowseFileDB import populate_database_from_browse_file

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def create_browse_file_from_fits_file_and_populate_database(
        db, fits_product_lid, collection_lidvid):
    # type: (BundleDB, str, str) -> int
    _create_browse_file_from_fits_file(fits_product_lid)
    browse_product_lid = None  # type: str
    basename = None  # type: unicode
    byte_size = None  # type: int
    populate_database_from_browse_file(
        db, browse_product_lid, collection_lidvid, basename, byte_size)
    return 0  # TODO fix this


def _create_browse_file_from_fits_file(fits_product_lid):
    # type: (str) -> None
    pass

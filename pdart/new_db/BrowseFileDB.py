from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB


def populate_database_from_browse_file(
        db, browse_product_lidvid, fits_product_lidvid, collection_lidvid,
        basename, byte_size):
    # type: (BundleDB, str, str, str, unicode, int) -> None
    db.create_browse_product(browse_product_lidvid,
                             fits_product_lidvid,
                             collection_lidvid)
    db.create_browse_file(basename, browse_product_lidvid, byte_size)

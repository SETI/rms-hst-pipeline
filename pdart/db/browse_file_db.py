from pdart.db.bundle_db import bundle_db


def populate_database_from_browse_file(
    db: bundle_db,
    browse_product_lidvid: str,
    fits_product_lidvid: str,
    collection_lidvid: str,
    os_filepath: str,
    basename: str,
    byte_size: int,
) -> None:
    db.create_browse_product(
        browse_product_lidvid, fits_product_lidvid, collection_lidvid
    )
    db.create_browse_file(os_filepath, basename, browse_product_lidvid, byte_size)

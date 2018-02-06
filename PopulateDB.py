"""
A script to create database files within the legacy archive sets.
"""
import os

from fs.path import join
from typing import TYPE_CHECKING

from pdart.new_db.BundleDB import create_bundle_db_from_os_filepath
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.pds4.Archives import get_any_archive
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID

if TYPE_CHECKING:
    from pdart.pds4.Bundle import Bundle


def wrap_bundle_and_populate_db(bundle):
    # type: (Bundle) -> None
    fp = bundle.absolute_filepath()
    db_filepath = join(fp, 'polynesia.db')
    try:
        os.remove(db_filepath)
    except:
        pass
    db = create_bundle_db_from_os_filepath(db_filepath)

    try:
        db.create_tables()
        vid = VID('1.0')
        bundle_lidvid = str(LIDVID.create_from_lid_and_vid(bundle.lid, vid))
        db.create_bundle(bundle_lidvid)
        print bundle_lidvid

        for collection in bundle.collections():
            collection_lidvid = str(
                LIDVID.create_from_lid_and_vid(collection.lid,
                                               vid))
            db.create_non_document_collection(collection_lidvid,
                                              bundle_lidvid)
            print '  ', collection_lidvid

            for product in collection.products():
                product_lidvid = str(
                    LIDVID.create_from_lid_and_vid(product.lid,
                                                   vid))
                db.create_fits_product(product_lidvid, collection_lidvid)
                # print '    ', product_lidvid

                for file in product.files():
                    filepath = file.full_filepath()
                    populate_database_from_fits_file(db,
                                                     filepath,
                                                     str(product_lidvid))
    except Exception as e:
        print '**** EXCEPTION: %s in %s' % (str(e), db_filepath)
        db.close()
        try:
            os.remove(db_filepath)
        except:
            pass
    else:
        db.close()


def run():
    archive = get_any_archive()
    for b in archive.bundles():
        wrap_bundle_and_populate_db(b)


if __name__ == '__main__':
    run()

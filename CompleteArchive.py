"""
SCRIPT: "Complete" is a verb: we are completing the archive by adding
labels, browse products, documentation, etc.
"""
from contextlib import closing
from typing import TYPE_CHECKING
import sqlite3

from pdart.pds4.Archives import get_any_archive
from pdart.pds4labels.DBCalls import *

if TYPE_CHECKING:
    from pdart.pds4.Archive import Archive
    from pdart.pds4.LID import LID


def make_browse_lid(fits_lid):
    # type: (LID) -> LID
    pass


def make_spice_lid(fits_lid):
    # type: (LID) -> LID
    pass


def make_documentation_lid(bundle_lid):
    # type: (LID) -> LID
    pass


def pre_make_fits_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_fits_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_fits_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_fits_database(conn, fits_lid, fits_file)
    # do something
    post_make_fits_database(conn, fits_lid, fits_file)


def pre_make_fits_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_fits_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_fits_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_fits_label(conn, fits_lid, fits_file)
    # do something
    post_make_fits_label(conn, fits_lid, fits_file)


def pre_make_browse_file(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_browse_file(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_browse_file(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_browse_file(conn, fits_lid, fits_file)
    # do something
    post_make_browse_file(conn, fits_lid, fits_file)


def pre_make_browse_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_browse_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_browse_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_browse_database(conn, fits_lid, fits_file)
    # do something
    post_make_browse_database(conn, fits_lid, fits_file)


def pre_make_browse_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_browse_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_browse_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_browse_label(conn, fits_lid, fits_file)
    # do something
    post_make_browse_label(conn, fits_lid, fits_file)


def pre_make_spice_file(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_spice_file(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_spice_file(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_spice_file(conn, fits_lid, fits_file)
    # do something
    post_make_spice_file(conn, fits_lid, fits_file)


def pre_make_spice_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_spice_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_spice_database(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_spice_database(conn, fits_lid, fits_file)
    # do something
    post_make_spice_database(conn, fits_lid, fits_file)


def pre_make_spice_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def post_make_spice_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pass


def make_spice_label(conn, fits_lid, fits_file):
    # type: (sqlite3.Connection, LID, unicode) -> None
    pre_make_spice_label(conn, fits_lid, fits_file)
    # do something
    post_make_spice_label(conn, fits_lid, fits_file)


def pre_make_documentation_files(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_documentation_files(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_documentation_files(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_documentation_files(conn, bundle_lid)
    # do something
    post_make_documentation_files(conn, bundle_lid)


def pre_make_documentation_database(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_documentation_database(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_documentation_database(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_documentation_database(conn, bundle_lid)
    # do something
    post_make_documentation_database(conn, bundle_lid)


def pre_make_documentation_label(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_documentation_label(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_documentation_label(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_documentation_label(conn, bundle_lid)
    # do something
    post_make_documentation_label(conn, bundle_lid)


def pre_make_collection_database(conn, collection_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_collection_database(conn, collection_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_collection_database(conn, collection_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_collection_database(conn, collection_lid)
    # do something
    post_make_collection_database(conn, collection_lid)


def pre_make_collection_inventory_and_label(conn, collection_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_collection_inventory_and_label(conn, collection_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_collection_inventory_and_label(conn, collection_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_collection_inventory_and_label(conn, collection_lid)
    # do something
    post_make_collection_inventory_and_label(conn, collection_lid)


def pre_make_bundle_database(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_bundle_database(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_bundle_database(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_bundle_database(conn, bundle_lid)
    # do something
    post_make_bundle_database(conn, bundle_lid)


def pre_make_bundle_label(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def post_make_bundle_label(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pass


def make_bundle_label(conn, bundle_lid):
    # type: (sqlite3.Connection, LID) -> None
    pre_make_bundle_label(conn, bundle_lid)
    # do something
    post_make_bundle_label(conn, bundle_lid)


def pre_complete_archive(archive):
    # type: (Archive) -> None
    pass


def post_complete_archive(archive):
    # type: (Archive) -> None
    pass


def complete_archive(archive):
    # type: (Archive) -> None
    # We're working with existing FITS files
    for bundle in archive.bundles():
        with closing(open_bundle_database(bundle)) as conn:
            for collection in bundle.collections():
                make_collection_database(conn, collection.lid)
                make_collection_inventory_and_label(conn, collection.lid)
                for product in collection.products():
                    for fits_file in product.files():
                        make_fits_database(conn, product.lid, fits_file)

                        # If we don't need out-of-line information,
                        # (i.e., information from other files), we can
                        # build the rest right here.

                        # Do I need the FITS file at all in the
                        # following calls or can I just run from the
                        # database?
                        make_fits_label(conn, product.lid, fits_file)

                        browse_lid = make_browse_lid(product.lid)
                        make_browse_file(conn, browse_lid, fits_file)
                        make_browse_database(conn, browse_lid, fits_file)
                        make_browse_label(conn, browse_lid, fits_file)

                        spice_lid = make_spice_lid(product.lid)
                        make_spice_file(conn, spice_lid, fits_file)
                        make_spice_database(conn, spice_lid, fits_file)
                        make_spice_label(conn, spice_lid, fits_file)

            documentation_lid = make_documentation_lid(bundle.lid)
            make_documentation_files(conn, documentation_lid)
            make_documentation_database(conn, documentation_lid)
            make_documentation_label(conn, documentation_lid)

            # make_bundle_files()  # Not needed
            make_bundle_database(conn, bundle.lid)
            make_bundle_label(conn, bundle.lid)


# We have multiple passes to do.  We may be building files, populating
# the database, or creating labels.  We may be working with FITS
# files, browse images, document collections, SPICE kernel products,
# or collections or bundles.  Each one should have pre- and
# post-conditions.

# Conceptually, we can run all of these, but we need to work
# bottom-up, since collections (and bundles) depend on their contents.

# But practically, we first suck up all the FITS info into the
# database because it's a lot cheaper to access than the filesystem,
# especially opening and parsing FITS files.  I.e., we do all the
# make_fits_db in one pass through the entire bundle for efficiency
# reasons (opening and closing the database).

# Do we do the same for other database actions?  Do make_browse_files
# and/or make_browse_db in one single bundle-wide pass?  TODO Figure
# this out.

# So, we end up with something like this:
#
# make_fits_files -- doesn't need to be done; they already exist
#
# make_fits_db -- yes, needs to be done
#
# make_fits_label -- yes, after the database
#
# ==== browse *PRODUCTS*, one for each raw product where raw means ...
#
# make_browse_files -- need to be created from each FITS file
#
# make_browse_db -- yes, perhaps at the same time as creating the
# files
#
# make_browse_label -- yes, build from the database
#
# ==== spice *PRODUCTS*, one for each spice-able product where
# spice-able means..
#
# make_spice_files -- need to be created from each FITS file
#
# make_spice_db -- yes, perhaps at the same time as creating the files
#
# make_spice_label -- yes, build from the database
#
# ==== collections (other than document) ==== make sure make_fits_db,
# make_browse_db, make_spice_db were already run.
#
# make_collection_files -- already exist from building products
#
# make_collection_db -- may have already been created by products
#
# make_collection_label -- yes, build from the database (inventory
# too)
#
# ==== document *COLLECTIONS* not products ====
#
# make_document_files -- need to be downloaded
#
# make_document_db -- yes, perhaps at the same time as creating the
# files
#
# make_document_label -- yes, build from the database
#
# ==== bundles  ====  make sure document collections were built
#
# make_bundle_files -- already exist
#
# make_bundle_db -- may have already been created by products
#
# make_bundle_label -- yes, build from the database

if __name__ == '__main__':
    archive = get_any_archive()
    complete_archive(archive)

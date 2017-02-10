"""
SCRIPT: "Complete" here is a verb: we are completing the archive by
adding labels, browse products, documentation, collections, bundles,
etc.
"""
import abc
from contextlib import closing
import io
import os
import os.path
import re
import shutil
import sqlite3
from typing import TYPE_CHECKING

from pdart.db.CompleteDatabase import *
from pdart.pds4.Archives import get_any_archive
from pdart.pds4.Product import Product
from pdart.pds4labels.FitsProductLabelXml import make_label
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES
from pdart.xml.Pretty import *
from pdart.xml.Schema import *
from pdart.xml.Templates import *

if TYPE_CHECKING:
    from pdart.pds4.Archive import Archive
    from pdart.pds4.File import File
    from pdart.pds4.LID import LID


def ensure_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def label_filepath(archive, lid):
    # type: (Archive, LID) -> unicode
    return Product(archive, lid).label_filepath()


def browse_image_filepath(archive, fits_lid):
    # type: (Archive, LID) -> unicode
    # TODO refactor
    fits_product = Product(archive, fits_lid)
    visit = fits_product.visit()

    raw_collection_filepath = fits_product.collection().absolute_filepath()
    # TODO Sloppy implementation: assumes only one 'data'
    browse_collection_filepath = \
        'browse'.join(re.split('data', raw_collection_filepath))

    basename = os.path.basename(fits_product.absolute_filepath())
    basename = os.path.splitext(basename)[0] + '.jpg'
    target_dir = os.path.join(browse_collection_filepath,
                              ('visit_%s' % visit))
    return os.path.join(target_dir, basename)


def make_fits_label(cursor, lid):
    # type: (sqlite3.Cursor, LID) -> unicode
    info = cursor.execute('SELECT * FROM fits_products WHERE lid=?',
                          (str(lid),))
    ian = '#### Investigation_Area_name ####'
    return unicode(pretty_print(make_label({
                    'lid': str(lid),
                    'proposal_id': '#### proposal_id ####',
                    'suffix': '#### suffix ####',
                    'file_name': '#### file_name ####',
                    'file_contents': combine_nodes_into_fragment([
                            interpret_text('#### file_contents ####')
                            ]),
                    'Investigation_Area_name': ian,
                    'investigation_lidvid': '#### investigation_lidvid ####',
                    'Observing_System': '#### Observing_System ####',
                    'Time_Coordinates': '#### Time_Coordinates ####',
                    'Target_Identification': '#### Target_Identification ####',
                    'HST': '#### HST ####',
                    }).toxml()))


def make_browse_label(cursor, lid):
    # type: (sqlite3.Cursor, LID) -> unicode
    # TODO implement
    return u'Ceci n\u2019est pas une \u00e9tiquette \u00e0 feuilleter.'


def make_browse_image(archive, cursor, fits_lid, fits_file):
    # type: (Archive, sqlite3.Cursor, LID, File) -> unicode
    # TODO implement
    filepath = browse_image_filepath(archive, fits_lid)
    ensure_dir(os.path.dirname(filepath))
    with io.open(filepath, 'w') as file:
        file.write(u'Ceci n\u2019est pas une image.')


def write_label(label, filepath):
    # type: (unicode, unicode) -> None
    with io.open(filepath, 'w') as file:
        file.write(label)


class Maker(object):
    """
    An abstract class for making things with lots of added checks.
    """
    __metaclass = abc.ABCMeta

    @abc.abstractmethod
    def check_requirements(self):
        # type: () -> None
        pass

    @abc.abstractmethod
    def check_results(self):
        # type: () -> None
        pass

    @abc.abstractmethod
    def make(self):
        # type: () -> None
        pass

    def __call__(self):
        # type: () -> None
        self.check_requirements()
        self.make()
        self.check_results()


class FileMaker(Maker):
    """
    An abstract class for putting files into a filesystem.
    """
    pass


class BrowseFileMaker(FileMaker):
    def __init__(self, archive, conn, fits_lid, browse_lid, fits_file):
        # type: (Archive, sqlite3.Connection, LID, LID, unicode) -> None
        self.archive = archive
        self.conn = conn
        self.fits_lid = fits_lid
        self.browse_lid = browse_lid
        self.fits_file = fits_file

    def check_requirements(self):
        # make sure the source FITS file exists
        assert os.path.isfile(self.fits_file.full_filepath())
        # make sure we're raw
        comp = self.fits_file.component
        assert isinstance(comp, Product)
        assert comp.collection().suffix() in RAW_SUFFIXES
        # TODO Anything else?

    def check_results(self):
        assert os.path.isfile(browse_image_filepath(archive, self.fits_lid))

    def make(self):
        with closing(self.conn.cursor()) as cursor:
            make_browse_image(self.archive, cursor,
                              self.fits_lid, self.fits_file)
            print 'Made browse image', self.browse_lid


class DatabaseRecordMaker(Maker):
    """
    An abstract class for making records in a SQLite database.
    """
    pass


class LabelMaker(Maker):
    """
    An abstract class for making PDS4 labels.
    """
    pass


class BrowseDatabaseRecordMaker(DatabaseRecordMaker):
    def __init__(self, conn, lid):
        # type: (sqlite3.Connection, LID) -> None
        self.conn = conn
        self.lid = lid

    def check_requirements(self):
        assert not exists_database_records_for_browse(self.conn, self.lid), \
            self.lid

    def check_results(self):
        # TODO records need to point up to collection.  That is
        # necessary because...
        assert exists_database_records_for_browse(self.conn, self.lid), \
            self.lid

    def make(self):
        with closing(self.conn.cursor()) as cursor:
            insert_browse_database_records(cursor, self.lid)
            print 'Inserted', self.lid


class FitsDatabaseRecordMaker(DatabaseRecordMaker):
    def __init__(self, conn, lid, file):
        # type: (sqlite3.Connection, LID, unicode) -> None
        self.conn = conn
        self.lid = lid

    def check_requirements(self):
        # I think these should vary for different kinds of DBs but
        # I'll implement them externally for now.
        assert not exists_database_records_for_fits(self.conn, self.lid), \
            self.lid

    def check_results(self):
        assert exists_database_records_for_fits(self.conn, self.lid), self.lid

    def make(self):
        with closing(self.conn.cursor()) as cursor:
            insert_fits_database_records(cursor, self.lid)
            print 'Inserted', self.lid


VERIFY = False


class FitsLabelMaker(LabelMaker):
    def __init__(self, archive, conn, lid, file):
        # type: (Archive, sqlite3.Connection, LID, unicode) -> None
        self.archive = archive
        self.conn = conn
        self.lid = lid

    def check_requirements(self):
        assert exists_database_records_for_fits(self.conn, self.lid), self.lid

    def check_results(self):
        fp = label_filepath(self.archive, self.lid)
        assert os.path.isfile(fp)
        if VERIFY:
            verify_label_or_raise_fp(fp)

    def make(self):
        with closing(self.conn.cursor()) as cursor:
            label = make_fits_label(cursor, self.lid)
            write_label(label, label_filepath(self.archive, self.lid))
            print 'Made label for', self.lid


class BrowseLabelMaker(LabelMaker):
    def __init__(self, archive, conn, lid, file):
        # type: (Archive, sqlite3.Connection, LID, unicode) -> None
        self.archive = archive
        self.conn = conn
        self.lid = lid

    def check_requirements(self):
        assert exists_database_records_for_browse(self.conn, self.lid), \
            self.lid

    def check_results(self):
        fp = label_filepath(self.archive, self.lid)
        assert os.path.isfile(fp)
        if VERIFY:
            verify_label_or_raise_fp(fp)

    def make(self):
        with closing(self.conn.cursor()) as cursor:
            label = make_browse_label(cursor, self.lid)
            write_label(label, label_filepath(self.archive, self.lid))
            print 'Made label for', self.lid


def make_browse_lid(fits_lid):
    # type: (LID) -> LID
    return fits_lid.to_browse_lid()


def make_spice_lid(fits_lid):
    # type: (LID) -> LID
    assert False, 'unimplemented'


def make_documentation_lid(bundle_lid):
    # type: (LID) -> LID
    pass


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
    # TODO Interesting idea, but is used nowhere.  Figure this out.
    pass


def post_complete_archive(archive):
    # type: (Archive) -> None
    # TODO Interesting idea, but is used nowhere.  Figure this out.
    pass


class SpiceFileMaker(FileMaker):
    def check_requirements(self):
        assert False, 'unimplemented'

    def check_results(self):
        assert False, 'unimplemented'

    def make(self):
        assert False, 'unimplemented'


class SpiceDatabaseRecordMaker(DatabaseRecordMaker):
    def check_requirements(self):
        assert False, 'unimplemented'

    def check_results(self):
        assert False, 'unimplemented'

    def make(self):
        assert False, 'unimplemented'


class SpiceLabelMaker(LabelMaker):
    def check_requirements(self):
        assert False, 'unimplemented'

    def check_results(self):
        assert False, 'unimplemented'

    def make(self):
        assert False, 'unimplemented'


def complete_archive(archive):
    # type: (Archive) -> None
    # We're working with existing FITS files

    # Set this True if you want to replace the databases with clean
    # ones.
    STARTING_CLEAN = True
    if STARTING_CLEAN:
        for collection in archive.collections():
            if collection.prefix() == 'browse':
                shutil.rmtree(collection.absolute_filepath())

    for bundle in archive.bundles():
        if STARTING_CLEAN:
            try:
                os.remove(bundle_database_filepath(bundle))
            except OSError:
                pass
        with closing(open_bundle_database(bundle)) as conn:
            for collection in bundle.collections():
                if collection.prefix() == 'data':
                    for product in collection.products():
                        for fits_file in product.files():
                            FitsDatabaseRecordMaker(conn,
                                                    product.lid,
                                                    fits_file)()

                            # If we don't need out-of-line information,
                            # (i.e., information from other files), we can
                            # build the rest right here.

                            # Do I need the FITS file at all in the
                            # following calls or can I just run from the
                            # database?
                            FitsLabelMaker(archive, conn,
                                           product.lid, fits_file)()

                            if collection.suffix() in RAW_SUFFIXES:
                                browse_lid = make_browse_lid(product.lid)
                                BrowseFileMaker(archive, conn,
                                                product.lid, browse_lid,
                                                fits_file)()

                                BrowseDatabaseRecordMaker(conn, browse_lid)()
                                BrowseLabelMaker(archive, conn,
                                                 browse_lid, fits_file)()

                            if False:
                                spice_lid = make_spice_lid(product.lid)
                                SpiceFileMaker()()
                                SpiceDatabaseRecordMaker()()
                                SpiceLabelMaker()()

                # TODO should check here that any new collections were
                # properly made.  The big loop body is, in effect, a
                # CollectionFileMaker call.  There's a pre (on what?)
                # that these collections don't yet exist.
                make_collection_database(conn, collection.lid)
                make_collection_inventory_and_label(conn, collection.lid)

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

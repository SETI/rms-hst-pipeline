import os
import re
import shutil
import tempfile

import fs.copy
import fs.path
import pdart.add_pds_tools
import picmaker  # need to precede this with 'import pdart.add_pds_tools'
from fs.osfs import OSFS
from typing import TYPE_CHECKING

from pdart.documents.Downloads import download_product_documents
from pdart.fs.DirUtils import lid_to_dir
from pdart.fs.V1FS import V1FS
from pdart.new_db.BundleDB import _BUNDLE_DB_NAME, \
    create_bundle_db_from_os_filepath
from pdart.new_db.BundleWalk import BundleWalk
from pdart.new_db.FitsFileDB import populate_database_from_fits_file
from pdart.new_labels.BrowseProductLabel import make_browse_product_label
from pdart.new_labels.BundleLabel import make_bundle_label
from pdart.new_labels.CollectionInventory import \
    get_collection_inventory_name, make_collection_inventory
from pdart.new_labels.CollectionLabel import get_collection_label_name, \
    make_collection_label
from pdart.new_labels.FitsProductLabel import make_fits_product_label
from pdart.pds4.HstFilename import HstFilename
from pdart.pds4.LID import LID
from pdart.pds4.LIDVID import LIDVID
from pdart.pds4.VID import VID
from pdart.pds4labels.RawSuffixes import RAW_SUFFIXES

if TYPE_CHECKING:
    from pdart.new_db.BundleDB import BundleDB
    from pdart.new_db.SqlAlchTables import BadFitsFile, BrowseFile, Bundle, \
        Collection, DocumentCollection, DocumentFile, FitsFile, FitsProduct, \
        NonDocumentCollection

_INITIAL_VID = VID('1.0')  # type: VID


def _browse_lidvid(lidvid):
    # type: (str) -> str
    raw_lidvid = LIDVID(lidvid)
    browse_lidvid = LIDVID.create_from_lid_and_vid(
        raw_lidvid.lid().to_browse_lid(),
        raw_lidvid.vid())
    return str(browse_lidvid)


def bundle_to_int(bundle_id):
    # type: (unicode) -> int
    """Converts a bundle_id to an int"""
    res = re.match(r'^hst_([0-9]{5})$', bundle_id)
    if res:
        return int(res.group(1))
    else:
        return None


def copy_files_from_download(src_dir, archive_dir):
    # type: (unicode, unicode) -> int
    """
    Copies the files from a MAST download directory to a new bundle
    directory.  The new archive uses a multi-version hierarchy to
    store the files.
    """
    src_fs = OSFS(src_dir)
    archive_fs = V1FS(archive_dir)
    for filepath in src_fs.walk.files(filter=['*.fits']):
        parts = fs.path.iteratepath(filepath)
        depth = len(parts)
        assert depth == 5, filepath
        bundle, _, _, product, filename = parts
        filename = filename.lower()
        hst_filename = HstFilename(filename)
        coll = 'data_%s_%s' % (hst_filename.instrument_name(),
                               hst_filename.suffix())
        new_path = fs.path.join(bundle, coll, product, filename)
        dirs, filename = fs.path.split(new_path)
        archive_fs.makedirs(dirs)
        fs.copy.copy_file(src_fs, filepath, archive_fs, new_path)
    if False:
        print 'Archive, seen as uni-versioned, looks like this:'
        archive_fs.tree()
        print 'Archive, seen as multi-versioned, looks like this:'
        OSFS(archive_dir).tree()
    return bundle_to_int(bundle)


def _create_lidvid_from_parts(parts):
    # type: (List[str]) -> str
    lid = LID.create_from_parts(parts)
    lidvid = LIDVID.create_from_lid_and_vid(lid, _INITIAL_VID)
    return str(lidvid)


def _bundle_dir(bundle_id, archive_dir):
    # type: (int, unicode) -> unicode
    bundle_name = 'hst_%05d' % bundle_id
    return fs.path.join(archive_dir, bundle_name)


def create_bundle_db(bundle_id, archive_dir):
    # type: (int, unicode) -> BundleDB
    """
    Creates a new, empty BundleDB in the bundle directory.  The
    database is written to hst_<bundle_id>/<bundle_db_name>.
    """
    bundle_dir = _bundle_dir(bundle_id, archive_dir)
    db = create_bundle_db_from_os_filepath(fs.path.join(bundle_dir,
                                                        _BUNDLE_DB_NAME))
    db.create_tables()

    bundle_name = 'hst_%05d' % bundle_id
    bundle_lidvid = _create_lidvid_from_parts([bundle_name])
    db.create_bundle(bundle_lidvid)
    return db


def populate_database(bundle_id, bundle_db, archive_dir):
    # type: (int, BundleDB, unicode) -> None
    """
    Populates the database with collections, products, and the FITS
    files' contents using the contents of the bundle directory in the
    given archive.
    """
    bundle = 'hst_%05d' % bundle_id
    bundle_lidvid = _create_lidvid_from_parts([bundle])
    v1fs = V1FS(archive_dir)
    for collection in v1fs.listdir(bundle):
        if '$' in collection:
            continue
        collection_path = fs.path.join(bundle, collection)
        collection_lidvid = _create_lidvid_from_parts([bundle, collection])
        bundle_db.create_non_document_collection(collection_lidvid,
                                                 bundle_lidvid)
        for product in v1fs.listdir(collection_path):
            if '$' in product:
                continue
            product_path = fs.path.join(collection_path, product)
            product_lidvid = _create_lidvid_from_parts(
                [bundle, collection, product])
            bundle_db.create_fits_product(product_lidvid, collection_lidvid)
            for fits_file in v1fs.listdir(product_path):
                if '$' in fits_file:
                    continue
                assert fs.path.splitext(fits_file)[1] == '.fits'
                fits_file_path = fs.path.join(product_path, fits_file)
                bundle_db.create_fits_product(product_lidvid,
                                              collection_lidvid)
                fits_sys_path = v1fs.getsyspath(fits_file_path)
                populate_database_from_fits_file(bundle_db, fits_sys_path,
                                                 product_lidvid)


def create_browse_products(bundle_id, bundle_db, archive_dir):
    # type: (int, BundleDB, unicode) -> None
    """
    Create browse products from appropriate products in the bundle.
    """
    archive_fs = V1FS(archive_dir)

    class _CreateBrowseProducts(BundleWalk):
        """
        A class for walking a bundle and building browse products (and
        collections) for image files.
        """

        def visit_non_document_collection(self, collection, post):
            # type: (NonDocumentCollection, bool) -> None
            if post:
                return
            # pre-visit
            if collection.suffix not in RAW_SUFFIXES:
                return
            collection_lidvid = str(collection.lidvid)
            browse_collection_lidvid = _browse_lidvid(collection_lidvid)
            bundle_lidvid = str(collection.bundle_lidvid)
            self.db.create_non_document_collection(browse_collection_lidvid,
                                                   bundle_lidvid)

        def visit_fits_product(self, fits_product, post):
            # type: (FitsProduct, bool) -> None
            if post:
                return
            collection_lidvid = str(fits_product.collection_lidvid)
            collection = self.db.get_collection(collection_lidvid)
            if collection.suffix not in RAW_SUFFIXES:
                return

            fits_product_lidvid = str(fits_product.lidvid)
            browse_collection_lidvid = _browse_lidvid(collection_lidvid)
            browse_product_lidvid = _browse_lidvid(fits_product_lidvid)
            self.db.create_browse_product(browse_product_lidvid,
                                          fits_product_lidvid,
                                          browse_collection_lidvid)
            file = self.db.get_product_file(fits_product_lidvid)
            # create browse file in the filesystem
            file_basename = file.basename
            browse_basename = (fs.path.splitext(file_basename)[0] +
                               '.jpg')

            fits_fs_filepath = fs.path.join(
                lid_to_dir(LIDVID(fits_product_lidvid).lid()),
                file_basename)
            browse_product_fs_dirpath = lid_to_dir(
                LIDVID(browse_product_lidvid).lid())

            fits_sys_filepath = archive_fs.getsyspath(fits_fs_filepath)
            archive_fs.makedirs(browse_product_fs_dirpath)
            browse_product_sys_dirpath = archive_fs.getsyspath(
                browse_product_fs_dirpath)

            # Picmaker expects a list of strings.  If you give it
            # unicode, it'll index into it and complain about '/'
            # not being a file.  So don't do that!
            picmaker.ImagesToPics([str(fits_sys_filepath)],
                                  browse_product_sys_dirpath,
                                  filter="None",
                                  percentiles=(1, 99))
            browse_sys_filepath = fs.path.join(
                browse_product_sys_dirpath, browse_basename)
            size = os.stat(browse_sys_filepath).st_size

            # create browse file record in the database
            self.db.create_browse_file(browse_basename,
                                       browse_product_lidvid,
                                       size)

    _CreateBrowseProducts(bundle_db).walk()


def _lidvid_to_dir(lidvid):
    # type: (str) -> unicode
    def get_lid(lidvid):
        # type: (str) -> LID
        return LIDVID(lidvid).lid()

    return lid_to_dir(get_lid(str(lidvid)))


def create_pds4_labels(bundle_id, bundle_db, archive_dir):
    # type: (int, BundleDB, unicode) -> None
    archive_fs = V1FS(archive_dir)

    class _CreateLabelsWalk(BundleWalk):
        """
        A class for walking a bundle and building labels (and a collection
        inventory) for it.
        """
        def visit_bundle(self, bundle, post):
            # type: (Bundle, bool) -> None
            if not post:
                return
            bundle_lidvid = str(bundle.lidvid)
            bundle_dir_path = _lidvid_to_dir(bundle_lidvid)
            label = make_bundle_label(self.db, False)
            label_filename = 'bundle.xml'
            label_filepath = fs.path.join(
                bundle_dir_path,
                label_filename)
            archive_fs.settext(label_filepath, unicode(label))

        def _post_visit_collection(self, collection):
            # type: (Collection) -> None
            """Common implementation for all collections."""
            collection_lidvid = str(collection.lidvid)
            collection_dir_path = _lidvid_to_dir(collection_lidvid)

            inventory = make_collection_inventory(self.db, collection_lidvid)
            inventory_filename = get_collection_inventory_name(
                self.db,
                collection_lidvid)
            inventory_filepath = fs.path.join(
                collection_dir_path,
                inventory_filename)
            archive_fs.settext(inventory_filepath, unicode(inventory))

            label = make_collection_label(self.db, collection_lidvid, False)
            label_filename = get_collection_label_name(self.db,
                                                       collection_lidvid)
            label_filepath = fs.path.join(
                collection_dir_path,
                label_filename)
            archive_fs.settext(label_filepath, unicode(label))

        def visit_document_collection(self, document_collection, post):
            # type: (DocumentCollection, bool) -> None
            if post:
                self._post_visit_collection(document_collection)

        def visit_non_document_collection(self, non_document_collection, post):
            # type: (NonDocumentCollection, bool) -> None
            if post:
                self._post_visit_collection(non_document_collection)

        def visit_browse_file(self, browse_file):
            # type: (BrowseFile) -> None
            label = make_browse_product_label(self.db,
                                              str(browse_file.product_lidvid),
                                              str(browse_file.basename),
                                              False)
            label_base = fs.path.splitext(browse_file.basename)[0]
            label_filename = label_base + '.xml'
            product_lidvid = str(browse_file.product_lidvid)
            product_dir_path = _lidvid_to_dir(product_lidvid)
            label_filepath = fs.path.join(
                product_dir_path,
                label_filename)
            archive_fs.settext(label_filepath, unicode(label))

        def visit_bad_fits_file(self, bad_fits_file):
            # type: (BadFitsFile) -> None
            assert False, (
                'Not yet handling bad FITS file %s in product %s' %
                (str(bad_fits_file.basename),
                 str(bad_fits_file.product_lidvid)))

        def visit_document_file(self, document_file):
            # type: (DocumentFile) -> None
            assert False, (
                'Not yet handling document file %s in product %s' %
                (str(document_file.basename),
                 str(document_file.product_lidvid)))

        def visit_fits_file(self, fits_file):
            # type: (FitsFile) -> None
            label = make_fits_product_label(self.db,
                                            str(fits_file.product_lidvid),
                                            str(fits_file.basename),
                                            False)
            label_base = fs.path.splitext(fits_file.basename)[0]
            label_filename = label_base + '.xml'
            product_lidvid = str(fits_file.product_lidvid)
            product_dir_path = _lidvid_to_dir(product_lidvid)
            label_filepath = fs.path.join(
                product_dir_path,
                label_filename)
            archive_fs.settext(label_filepath, unicode(label))

    _CreateLabelsWalk(bundle_db).walk()


def create_document_collection(bundle_id, bundle_db, archive_dir,
                               document_dir, document_files):
    # type: (int, BundleDB, unicode, unicode, Set[unicode]) -> None
    if not document_files:
        return
    archive_fs = V1FS(archive_dir)
    bundle_part = 'hst_%05d' % bundle_id
    bundle_lidvid = _create_lidvid_from_parts([bundle_part])
    collection_lidvid = _create_lidvid_from_parts([bundle_part,
                                                   'document'])

    # create the collection in the database
    bundle_db.create_document_collection(collection_lidvid, bundle_lidvid)

    # create the product in the database
    product_lidvid = _create_lidvid_from_parts([bundle_part,
                                                'document',
                                                'phase2'])

    bundle_db.create_document_product(product_lidvid, collection_lidvid)

    # create the files in the database
    for basename in document_files:
        bundle_db.create_document_file(basename, product_lidvid)

    # create the collection in the filesystem
    document_collection_dir = _lidvid_to_dir(collection_lidvid)
    archive_fs.makedirs(document_collection_dir)

    # create the product in the filesystem
    document_product_dir = _lidvid_to_dir(product_lidvid)
    archive_fs.makedirs(document_product_dir)

    # create the files in the filesystem
    doc_fs = OSFS(document_dir)
    fs.copy.copy_dir(doc_fs, u'/', archive_fs, document_product_dir)


def start_bundle(src_dir, archive_dir):
    # type: (unicode, unicode) -> None
    """
    Create the first version of a bundle from a MAST download.
    """
    bundle_id = copy_files_from_download(src_dir, archive_dir)
    bundle_db = create_bundle_db(bundle_id, archive_dir)
    populate_database(bundle_id, bundle_db, archive_dir)
    create_browse_products(bundle_id, bundle_db, archive_dir)
    # create_spice_products(bundle_id, bundle_db, archive_dir)

    documents_dir = tempfile.mkdtemp()
    try:
        docs = download_product_documents(bundle_id, documents_dir)
        create_document_collection(bundle_id, bundle_db, archive_dir,
                                   documents_dir, docs)
    finally:
        shutil.rmtree(documents_dir)
    create_pds4_labels(bundle_id, bundle_db, archive_dir)

    # TODO: more to do.  Make SPICE collections, document collections,
    # and labels.

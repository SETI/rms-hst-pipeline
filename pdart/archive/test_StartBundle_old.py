import os.path
import tempfile
import unittest

import fs.path

from pdart.archive.StartBundle_old import *
from pdart.archive.StartBundle_old import _INITIAL_VID
from pdart.pds4.LIDVID import LIDVID


def _path_to_testfiles():
    # type: () -> unicode
    """Return the path to files needed for testing."""
    return os.path.join(os.path.dirname(__file__), 'testfiles')


def _list_rel_filepaths(root_dir):
    # type: (unicode) -> List[unicode]
    def _list_rel_filepaths_gen():
        for (dirpath, _, filenames) in os.walk(root_dir):
            rel_dirpath = fs.path.relativefrom(root_dir, dirpath)
            for filename in filenames:
                yield os.path.join(rel_dirpath, filename)

    return sorted(_list_rel_filepaths_gen())


class TestStartBundle(unittest.TestCase):
    def setUp(self):
        self.base_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base_directory)

    def test_create_bundle_dir(self):
        # type: () -> None

        # make one; test that it creates its directory
        bundle_dir = os.path.join(self.base_directory, 'hst_01234')
        self.assertFalse(os.path.isdir(bundle_dir))
        create_bundle_dir(1234, self.base_directory)
        self.assertTrue(os.path.isdir(bundle_dir))

        # make another; test that it creates its directory
        bundle_dir = os.path.join(self.base_directory, 'hst_00000')
        self.assertFalse(os.path.isdir(bundle_dir))
        create_bundle_dir(0, self.base_directory)
        self.assertTrue(os.path.isdir(bundle_dir))

        # re-making does nothing and is harmless
        bundle_dir = os.path.join(self.base_directory, 'hst_00000')
        self.assertTrue(os.path.isdir(bundle_dir))
        create_bundle_dir(0, self.base_directory)
        self.assertTrue(os.path.isdir(bundle_dir))

        # raises an exception if its a file: just shouldn't happen
        bundle_dir = os.path.join(self.base_directory, 'hst_00666')
        self.assertFalse(os.path.isdir(bundle_dir))
        self.assertFalse(os.path.isfile(bundle_dir))
        with open(bundle_dir, 'w') as f:
            f.write('xxx')

        self.assertTrue(os.path.isfile(bundle_dir))
        with self.assertRaises(Exception):
            create_bundle_dir(666, self.base_directory)

    def test_create_bundle_version_view(self):
        # type: () -> None
        bundle_view, bundle_db = \
            create_bundle_version_view_and_db(13012,
                                              self.base_directory)
        try:
            self.assertTrue(bundle_view)
            self.assertTrue(bundle_db)
            self.assertTrue(os.path.isdir(os.path.join(self.base_directory,
                                                       'hst_13012')))
            fs.osfs.OSFS(self.base_directory).tree()
            self.assertEquals([u'hst_13012/bundle$database.db',
                               u'hst_13012/v$1.0/subdir$versions.txt'],
                              _list_rel_filepaths(self.base_directory))
        finally:
            bundle_db.close()

    def test_create_bundle_db(self):
        # type: () -> None
        bundle_id = 12345
        create_bundle_dir(bundle_id, self.base_directory)
        db = create_bundle_db(bundle_id, self.base_directory)
        try:
            # returns the DB
            self.assertTrue(db)
            db_filename = os.path.join(self.base_directory,
                                       'hst_12345',
                                       'bundle$database.db')
            # creates the DB file
            self.assertTrue(os.path.isfile(db_filename))
            self.assertEquals([
                u'hst_12345/bundle$database.db'
            ], _list_rel_filepaths(self.base_directory))

            bundle_lid = LID.create_from_parts(['hst_12345'])
            bundle_lidvid = LIDVID.create_from_lid_and_vid(
                bundle_lid,
                _INITIAL_VID)
            self.assertTrue(db.bundle_exists(str(bundle_lidvid)))
        finally:
            db.close()

    def test_copy_downloaded_files(self):
        # type: () -> None
        bundle_id = 13012
        create_bundle_dir(bundle_id, self.base_directory)
        bundle_db = create_bundle_db(bundle_id, self.base_directory)

        try:
            copy_downloaded_files(bundle_db, bundle_id,
                                  _path_to_testfiles(), self.base_directory)

            # Make sure the files arrived in the right places in the
            # filesystem.
            self.assertEquals([
                u'hst_13012/bundle$database.db',
                u'hst_13012/data_acs_drz/jbz504010/v$1.0/jbz504011_drz.fits',
                u'hst_13012/data_acs_drz/jbz504020/v$1.0/jbz504021_drz.fits',
                u'hst_13012/data_acs_drz/jbz504eoq/v$1.0/jbz504eoq_drz.fits',
                u'hst_13012/data_acs_flt/jbz504eoq/v$1.0/jbz504eoq_flt.fits'],
                _list_rel_filepaths(self.base_directory))

            # Make sure the data ended up in the database.

            # the bundle
            bundle_lidvid = 'urn:nasa:pds:hst_13012::1.0'
            self.assertTrue(bundle_db.bundle_exists(bundle_lidvid))

            # the collections
            collection_lidvids = [
                'urn:nasa:pds:hst_13012:data_acs_drz::1.0',
                'urn:nasa:pds:hst_13012:data_acs_flt::1.0']
            for collection_lidvid in collection_lidvids:
                self.assertTrue(bundle_db.non_document_collection_exists(
                    collection_lidvid),
                    msg=collection_lidvid)

            # the products
            product_lidvids = [
                'urn:nasa:pds:hst_13012:data_acs_drz:jbz504010::1.0',
                'urn:nasa:pds:hst_13012:data_acs_drz:jbz504020::1.0',
                'urn:nasa:pds:hst_13012:data_acs_drz:jbz504eoq::1.0',
                'urn:nasa:pds:hst_13012:data_acs_flt:jbz504eoq::1.0']
            for product_lidvid in product_lidvids:
                self.assertTrue(bundle_db.fits_product_exists(product_lidvid),
                                msg=product_lidvid)

            # the FITS files
            fits_files = [
                ('jbz504011_drz.fits',
                 'urn:nasa:pds:hst_13012:data_acs_drz:jbz504010::1.0'),
                ('jbz504021_drz.fits',
                 'urn:nasa:pds:hst_13012:data_acs_drz:jbz504020::1.0'),
                ('jbz504eoq_drz.fits',
                 'urn:nasa:pds:hst_13012:data_acs_drz:jbz504eoq::1.0'),
                ('jbz504eoq_flt.fits',
                 'urn:nasa:pds:hst_13012:data_acs_flt:jbz504eoq::1.0')]
            # NOTE: This bundle is anomalous; the filenames don't
            # always match the product names.  It's not a bug in the
            # code.
            for basename, product_lidvid in fits_files:
                self.assertTrue(bundle_db.fits_file_exists(basename,
                                                           product_lidvid),
                                msg=basename)
                # assume that if the first HDU got in, they all did
                self.assertTrue(bundle_db.hdu_exists(0,
                                                     basename,
                                                     product_lidvid),
                                msg=basename)
        finally:
            bundle_db.close()

    def test_make_browse_collections(self):
        bundle_id = 13012
        create_bundle_dir(bundle_id, self.base_directory)
        bundle_db = create_bundle_db(bundle_id, self.base_directory)
        try:
            copy_downloaded_files(bundle_db, bundle_id,
                                  _path_to_testfiles(), self.base_directory)
            make_browse_collections(bundle_db, bundle_id, self.base_directory)

            # Make sure the files arrived in the right places in the
            # filesystem.
            self.assertEquals([
                u'hst_13012/browse_acs_flt/' +
                u'jbz504eoq/v$1.0/jbz504eoq_flt.jpg',
                u'hst_13012/bundle$database.db',
                u'hst_13012/data_acs_drz/jbz504010/v$1.0/jbz504011_drz.fits',
                u'hst_13012/data_acs_drz/jbz504020/v$1.0/jbz504021_drz.fits',
                u'hst_13012/data_acs_drz/jbz504eoq/v$1.0/jbz504eoq_drz.fits',
                u'hst_13012/data_acs_flt/jbz504eoq/v$1.0/jbz504eoq_flt.fits'],
                _list_rel_filepaths(self.base_directory))

            # Make sure the data ended up in the database.

            # In the collection
            collection_lidvid = 'urn:nasa:pds:hst_13012:browse_acs_flt::1.0'
            self.assertTrue(bundle_db.non_document_collection_exists(
                collection_lidvid), msg=collection_lidvid)

            # In the product
            product_lidvid = \
                'urn:nasa:pds:hst_13012:browse_acs_flt:jbz504eoq::1.0'
            self.assertTrue(bundle_db.browse_product_exists(
                product_lidvid), msg=product_lidvid)

            # In the file
            self.assertTrue(bundle_db.browse_file_exists(
                'jbz504eoq_flt.jpg',
                product_lidvid), msg=product_lidvid)

        finally:
            bundle_db.close()

    @unittest.skip('unimplemented')
    def test_make_document_collection(self):
        bundle_id = 13012
        create_bundle_dir(bundle_id, self.base_directory)
        bundle_db = create_bundle_db(bundle_id, self.base_directory)
        try:
            copy_downloaded_files(bundle_db, bundle_id,
                                  _path_to_testfiles(), self.base_directory)
            make_browse_collections(bundle_db, bundle_id, self.base_directory)
            make_document_collection(bundle_db, bundle_id)

            # Make sure the files arrived in the right places in the
            # filesystem.
            self.assertEquals([
                u'hst_13012/browse_acs_flt/' +
                u'jbz504eoq/v$1.0/jbz504eoq_flt.jpg',
                u'hst_13012/bundle$database.db',
                u'hst_13012/data_acs_drz/jbz504010/v$1.0/jbz504011_drz.fits',
                u'hst_13012/data_acs_drz/jbz504020/v$1.0/jbz504021_drz.fits',
                u'hst_13012/data_acs_drz/jbz504eoq/v$1.0/jbz504eoq_drz.fits',
                u'hst_13012/data_acs_flt/jbz504eoq/v$1.0/jbz504eoq_flt.fits',
                u'hst_13012/document/v$1.0/something.txt'],
                _list_rel_filepaths(self.base_directory))

            # Make sure the data ended up in the database.

            # the collection
            collection_lidvid = 'urn:nasa:pds:hst_13012:document'
            self.assertTrue(bundle_db.document_collection_exists(
                collection_lidvid))

            # the document files
            files = []
            for file in files:
                # TODO I have no tests for files?  What does the
                # label-making code do?
                pass  # self.assertTrue(bundle_db.document_file_exists...)

        finally:
            bundle_db.close()

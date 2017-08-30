import abc
from typing import TYPE_CHECKING
from fs.path import basename, join

from pdart.fs.SubdirVersions import readSubdirVersions
from pdart.fs.VersionedFS import ALL_PATS, ROOT, SUBDIR_VERSIONS_FILENAME, \
    VERSION_DIR_PATS

if TYPE_CHECKING:
    from typing import List


class VersionedViewTestCases(object):
    """
    A set of test cases that should hold for any VersionedView, that
    is, any filesystem organized to include versioning.
    """
    def make_fs(self):
        raise NotImplementedError('implement me')

    def setUp(self):
        self.view = self.make_fs()

    def destroy_fs(self, fs):
        fs.close()

    def tearDown(self):
        self.destroy_fs(self.view)

    def check_bundle_dir(self, bundle_dir):
        # type: (unicode) -> None

        # Everything under the bundle dir must also be a dir.  There
        # are version directories and collection directories.
        files = [info.name
                 for info in self.view.filterdir(bundle_dir,
                                                 ALL_PATS, None,
                                                 ALL_PATS, None)]

        # No files in the bundle dir
        self.assertFalse(files)

        bundle_versions = [info.name
                           for info in self.view.filterdir(bundle_dir,
                                                           None,
                                                           VERSION_DIR_PATS,
                                                           None,
                                                           ALL_PATS)]
        # at least one version dir
        self.assertTrue(len(bundle_versions) >= 1)

        # version and collection dirs are all
        collections = [info.name
                       for info
                       in self.view.filterdir(bundle_dir,
                                              None, ALL_PATS,
                                              VERSION_DIR_PATS, None)]
        self.assertEqual(set(self.view.listdir(bundle_dir)),
                         set(bundle_versions + collections))

        for version in bundle_versions:
            VERSION_DIR = join(bundle_dir, version)
            self.check_version_dir(VERSION_DIR)

        for collection in collections:
            COLLECTION_DIR = join(bundle_dir, collection)
            self.check_collection_dir(COLLECTION_DIR)

    def check_collection_dir(self, collection_dir):
        # type: (unicode) -> None

        # Everything under the collection dir must also be a dir.  There
        # are version directories and product directories.
        files = [info.name
                 for info in self.view.filterdir(collection_dir,
                                                 ALL_PATS, None,
                                                 ALL_PATS, None)]

        # No files in the collection dir
        self.assertFalse(files)

        collection_versions = [info.name
                               for info
                               in self.view.filterdir(collection_dir,
                                                      None, VERSION_DIR_PATS,
                                                      None, ALL_PATS)]
        # at least one version dir
        self.assertTrue(len(collection_versions) >= 1)

        # version and collection dirs are all
        products = [info.name
                    for info in self.view.filterdir(collection_dir,
                                                    None, ALL_PATS,
                                                    VERSION_DIR_PATS, None)]
        self.assertEqual(set(self.view.listdir(collection_dir)),
                         set(collection_versions + products))

        for version in collection_versions:
            VERSION_DIR = join(collection_dir, version)
            self.check_version_dir(VERSION_DIR)

        for product in products:
            PRODUCT_DIR = join(collection_dir, product)
            self.check_product_dir(PRODUCT_DIR)

    def check_product_dir(self, product_dir):
        # type: (unicode) -> None

        # Everything under the product dir must also be a dir.  There
        # are version directories and product directories.
        files = [info.name
                 for info in self.view.filterdir(product_dir,
                                                 ALL_PATS, None,
                                                 ALL_PATS, None)]

        # No files in the product dir
        self.assertFalse(files)

        product_versions = [info.name
                            for info
                            in self.view.filterdir(product_dir,
                                                   None, VERSION_DIR_PATS,
                                                   None, ALL_PATS)]
        # at least one version dir
        self.assertTrue(len(product_versions) >= 1)

        # version dirs are all
        self.assertEqual(set(self.view.listdir(product_dir)),
                         set(product_versions))

        for version in product_versions:
            VERSION_DIR = join(product_dir, version)
            self.check_version_dir(VERSION_DIR)

    def check_version_dir(self, version_dir):
        # type: (unicode) -> None

        assert basename(version_dir)[0:2] == 'v$'
        # All version dirs must contain a subdir versions file
        SUBDIR_VERSIONS_FILEPATH = join(version_dir,
                                        SUBDIR_VERSIONS_FILENAME)

        self.assertTrue(self.view.exists(SUBDIR_VERSIONS_FILEPATH))
        self.assertTrue(self.view.isfile(SUBDIR_VERSIONS_FILEPATH))

        # check that the subdir$versions.txt file is in the right
        # format
        self.check_subdir_versions_file(version_dir)

        # version dirs contain only files
        self.assertFalse(list(self.view.filterdir(version_dir,
                                                  None, None,
                                                  VERSION_DIR_PATS, ALL_PATS)))

    def check_subdir_versions_file(self,
                                   version_dir):
        d = readSubdirVersions(self.view, version_dir)
        for subdir_name, version in d.items():
            # each subdirectory entry must correspond to an
            # existing subdirectory
            subdir = join(version_dir, '..', subdir_name, 'v$' + version)
            self.view.exists(subdir)
            self.view.isdir(subdir)

    def test_has_bundle_dirs(self):
        self.view.isdir(ROOT)
        # There is only one directory under root, corresponding to the
        # bundle.
        self.assertEquals(1, len(self.view.listdir(ROOT)))
        BUNDLE_NAME = self.view.listdir(ROOT)[0]
        BUNDLE_DIR = join(ROOT, BUNDLE_NAME)
        self.check_bundle_dir(BUNDLE_DIR)

    # We implement unittest.TestCase's methods conditionally, only to
    # keep mypy happy.
    if TYPE_CHECKING:
        def assertTrue(self, cond, msg=None):
            raise NotImplementedError('implement me')

        def assertFalse(self, cond, msg=None):
            raise NotImplementedError('implement me')

        def assertEqual(self, lhs, rhs, msg=None):
            raise NotImplementedError('implement me')

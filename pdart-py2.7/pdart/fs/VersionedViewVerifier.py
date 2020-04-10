"""
Functionality to verify a versioned filesystem.
"""
from fs.path import basename, join
from typing import TYPE_CHECKING

from pdart.fs.SubdirVersions import read_subdir_versions_from_directory
from pdart.fs.VersionedFS import ROOT, SUBDIR_VERSIONS_FILENAME, scan_vfs_dir

if TYPE_CHECKING:
    from pdart.fs.OldVersionView import OldVersionView


class VersionedViewException(Exception):
    def __init__(self, value):
        Exception.__init__(self, value)


class VersionedViewVerifier(object):
    """
    Verification for a VersionedView, that is, any filesystem
    organized to include versioning.
    """

    def __init__(self, view):
        # type: (OldVersionView) -> None
        self.view = view
        self.test_has_bundle_dirs()

    def check_bundle_dir(self, bundle_dir):
        # type: (unicode) -> None
        """
        Verify a bundle directory.
        """

        # Everything under the bundle dir must also be a dir.  There
        # are version directories and collection directories.
        (
            ordinary_file_infos,
            ordinary_dir_infos,
            subdir_versions_file_infos,
            version_dir_infos,
        ) = scan_vfs_dir(self.view, bundle_dir)

        # No files in the bundle dir
        self.assertFalse(ordinary_file_infos + subdir_versions_file_infos)

        # At least one version dir
        self.assertTrue(len(version_dir_infos) >= 1)

        # Version and collection dirs are all that's in the bundle dir
        self.assertEqual(
            [coll_dir_info.name for coll_dir_info in ordinary_dir_infos]
            + [version_dir_info.name for version_dir_info in version_dir_infos],
            self.view.listdir(bundle_dir),
        )

        for version_info in version_dir_infos:
            VERSION_DIR = join(bundle_dir, version_info.name)
            self.check_version_dir(VERSION_DIR)

        for collection_info in ordinary_dir_infos:
            COLLECTION_DIR = join(bundle_dir, collection_info.name)
            self.check_collection_dir(COLLECTION_DIR)

    def check_collection_dir(self, collection_dir):
        # type: (unicode) -> None
        """
        Verify a collection directory.
        """
        # Everything under the collection dir must also be a dir.  There
        # are version directories and product directories.
        (
            ordinary_file_infos,
            ordinary_dir_infos,
            subdir_versions_file_infos,
            version_dir_infos,
        ) = scan_vfs_dir(self.view, collection_dir)

        # No files in the collection dir
        self.assertFalse(ordinary_file_infos + subdir_versions_file_infos)

        # at least one version dir
        collection_versions = [info.name for info in version_dir_infos]
        self.assertTrue(len(collection_versions) >= 1)

        # Product and version dirs are all
        self.assertEqual(
            [prod_dir_info.name for prod_dir_info in ordinary_dir_infos]
            + [version_dir_info.name for version_dir_info in version_dir_infos],
            self.view.listdir(collection_dir),
        )

        for version_info in version_dir_infos:
            VERSION_DIR = join(collection_dir, version_info.name)
            self.check_version_dir(VERSION_DIR)

        for product_info in ordinary_dir_infos:
            PRODUCT_DIR = join(collection_dir, product_info.name)
            self.check_product_dir(PRODUCT_DIR)

    def check_product_dir(self, product_dir):
        # type: (unicode) -> None
        """
        Verify a product directory.
        """

        # Everything under the product dir must also be a dir.  There
        # are version directories and product directories.
        (
            ordinary_file_infos,
            ordinary_dir_infos,
            subdir_versions_file_infos,
            version_dir_infos,
        ) = scan_vfs_dir(self.view, product_dir)

        # No files in the product dir
        self.assertFalse(ordinary_file_infos + subdir_versions_file_infos)

        # at least one version dir
        product_versions = [info.name for info in version_dir_infos]
        self.assertTrue(len(product_versions) >= 1)

        # Version dirs are all
        self.assertEqual(
            set([version_dir_info.name for version_dir_info in version_dir_infos]),
            set(self.view.listdir(product_dir)),
        )

        for version_info in version_dir_infos:
            VERSION_DIR = join(product_dir, version_info.name)
            self.check_version_dir(VERSION_DIR)

    def check_version_dir(self, version_dir):
        # type: (unicode) -> None
        """
        Verify a version directory.
        """

        assert basename(version_dir)[0:2] == "v$"
        # All version dirs must contain a subdir versions file
        SUBDIR_VERSIONS_FILEPATH = join(version_dir, SUBDIR_VERSIONS_FILENAME)

        self.assertTrue(self.view.exists(SUBDIR_VERSIONS_FILEPATH))
        self.assertTrue(self.view.isfile(SUBDIR_VERSIONS_FILEPATH))

        # check that the subdir$versions.txt file is in the right
        # format
        self.check_subdir_versions_file(version_dir)

        # version dirs contain only files
        (
            ordinary_file_infos,
            ordinary_dir_infos,
            subdir_versions_file_infos,
            version_dir_infos,
        ) = scan_vfs_dir(self.view, version_dir)
        self.assertFalse(ordinary_dir_infos + version_dir_infos)

    def check_subdir_versions_file(self, version_dir):
        # type: (unicode) -> None
        """
        Verify a subdir-versions file.
        """
        d = read_subdir_versions_from_directory(self.view, version_dir)
        for subdir_name, version in d.items():
            # each subdirectory entry must correspond to an
            # existing subdirectory
            subdir = join(version_dir, "..", subdir_name, "v$" + version)
            self.view.exists(subdir)
            self.view.isdir(subdir)

    def test_has_bundle_dirs(self):
        # type: () -> None
        """
        Test that the filesystem has a single directory corresponding
        to the bundle.
        """
        self.view.isdir(ROOT)
        # There is only one directory under root, corresponding to the
        # bundle.
        self.assertEqual(1, len(self.view.listdir(ROOT)))
        BUNDLE_NAME = self.view.listdir(ROOT)[0]
        BUNDLE_DIR = join(ROOT, BUNDLE_NAME)
        self.check_bundle_dir(BUNDLE_DIR)

    def assertTrue(self, cond, msg=None):
        """
        Raise an exception with the given message if the first argument
        is not truthy.
        """
        if not cond:
            raise VersionedViewException(msg)

    def assertFalse(self, cond, msg=None):
        """
        Raise an exception with the given message if the first argument
        is not falsey.
        """
        if cond:
            raise VersionedViewException(msg)

    def assertEqual(self, lhs, rhs, msg=None):
        """
        Raise an exception with the given message if the two arguments
        are not equal.
        """
        cond = lhs == rhs
        if not cond:
            msg2 = "%s != %s" % (lhs, rhs)
            if msg:
                val = (msg, msg2)
            else:
                val = msg2
            raise VersionedViewException(val)

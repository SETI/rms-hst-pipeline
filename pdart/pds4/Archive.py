"""Representation of an archive containing PDS4 bundles."""

from fs.osfs import OSFS
from typing import TYPE_CHECKING

from pdart.pds4.Bundle import *
from pdart.pds4.Collection import *
from pdart.pds4.LID import *
from pdart.pds4.Product import *

if TYPE_CHECKING:
    from typing import Iterator


class Archive(object):
    """An :class:`~pdart.pds4.Archive` containing PDS4 Bundles."""

    def __init__(self, root):
        # type: (unicode) -> None
        """
        Create an :class:`~pdart.pds4.Archive` given a filepath to an existing
        directory.
        """
        self.root = root
        self.root_fs = OSFS(self.root)

    def __eq__(self, other):
        if not isinstance(other, Archive):
            return False
        return self.root == other.root

    def __str__(self):
        return repr(self.root)

    def __repr__(self):
        return 'Archive(%r)' % self.root

    # Verifying parts

    @staticmethod
    def is_valid_instrument(inst):
        # type: (str) -> bool
        """Return True iff the argument is a valid instrument name."""
        return inst in ['acs', 'wfc3', 'wfpc2']

    @staticmethod
    def is_valid_proposal(prop):
        # type: (int) -> bool
        """Return True iff the argument is a valid integer HST proposal ID."""
        return isinstance(prop, int) and 0 <= prop and prop <= 99999

    @staticmethod
    def is_valid_visit(vis):
        # type: (str) -> bool
        """Return True iff the argument is a valid visit ID."""
        try:
            return re.match(r'\A[a-z0-9][a-z0-9]\Z', vis) is not None
        except Exception:
            return False

    @staticmethod
    def is_valid_bundle_dir_basename(dirname):
        # type: (unicode) -> bool
        """Return True iff the argument is a valid bundle directory name."""
        return re.match(Bundle.DIRECTORY_PATTERN, dirname) is not None

    @staticmethod
    def is_valid_collection_dir_basename(dirname):
        # type: (unicode) -> bool
        """
        Return True iff the argument is a valid collection directory
        name.
        """
        return re.match(Collection.DIRECTORY_PATTERN, dirname) is not None

    @staticmethod
    def is_valid_product_dir_basename(dirname):
        # type: (unicode) -> bool
        """
        Return True iff the argument is a valid product visit
        directory name.
        """
        return re.match(Product.VISIT_DIRECTORY_PATTERN, dirname) is not None

    @staticmethod
    def is_hidden_file_basename(basename):
        # type: (unicode) -> bool
        """
        Return True iff the file is a hidden file.  Approximated by
        checking if its first character is a dot (Unix hidden-file
        convention).
        """
        return basename[0] == '.'

    # Walking the hierarchy with objects
    def bundles(self):
        # type: () -> Iterator[Bundle]
        """Generate the bundles stored in this :class:`~pdart.pds4.Archive`."""
        for subdir in self.root_fs.listdir(u'/'):
            if re.match(Bundle.DIRECTORY_PATTERN, subdir):
                bundle_lid = LID('urn:nasa:pds:%s' % subdir)
                yield Bundle(self, bundle_lid)

    def collections(self):
        # type: () -> Iterator[Collection]
        """
        Generate the collections stored in this
        :class:`~pdart.pds4.Archive`.
        """
        for b in self.bundles():
            for c in b.collections():
                yield c

    def products(self):
        # type: () -> Iterator[Product]
        """
        Generate the products stored in this
        :class:`~pdart.pds4.Archive`.
        """
        for b in self.bundles():
            for c in b.collections():
                for p in c.products():
                    yield p

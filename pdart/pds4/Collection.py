"""Representation of a PDS4 collection."""
import os.path
import re

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.

from pdart.pds4.Component import *
from pdart.pds4.LID import *

import pdart.pds4.Bundle  # for mypy
import pdart.pds4.Product  # for mypy
from typing import Iterator  # for mypy


class Collection(Component):
    """A PDS4 Collection."""

    DIRECTORY_PATTERN = r'\A([a-z]+)_([a-z0-9]+)_([a-z0-9_]+)\Z'
    # type: str

    """
    The collection's directory name is of the form
    '<prefix>_<instrument>_<suffix>'.  This pattern may be used to
    validate directory names or to extract fields from it.
    """

    def __init__(self, arch, lid):
        # type: (pdart.pds4.Archive.Archive, LID) -> None
        """
        Create a :class:`~pdart.pds4.Collection` given the
        :class:`~pdart.pds4.Archive` it lives in and its
        :class:`~pdart.pds4.LID`.
        """
        assert lid.is_collection_lid()
        super(Collection, self).__init__(arch, lid)

    def __repr__(self):
        return 'Collection(%r, %r)' % (self.archive, self.lid)

    def instrument(self):
        # type: () -> unicode
        """
        Return the instrument for this
        :class:`~pdart.pds4.Collection`.  It is calculated from the
        collection's :class:`~pdart.pds4.LID`.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(2)

    def prefix(self):
        # type: () -> unicode
        """
        Return the prefix for FITS files in this
        :class:`~pdart.pds4.Collection`.  It is calculated from the
        collection's :class:`~pdart.pds4.LID`.

        'data' and 'browse' are some possible results.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(1)

    def suffix(self):
        # type: () -> unicode
        """
        Return the suffix for FITS files in this
        :class:`~pdart.pds4.Collection`.  It is calculated from the
        collection's :class:`~pdart.pds4.LID`.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(3)

    def absolute_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the collection's directory."""
        return os.path.join(self.archive.root,
                            self.lid.bundle_id, self.lid.collection_id)

    def label_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the collection's label."""
        collection_fp = self.absolute_filepath()
        name = 'collection_%s.xml' % self.prefix()
        return os.path.join(collection_fp, name)

    def inventory_name(self):
        # type: () -> unicode
        """Return the filename of the collection's inventory."""
        return 'collection_%s.csv' % self.prefix()

    def inventory_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the collection's inventory."""
        return os.path.join(self.absolute_filepath(), self.inventory_name())

    def products(self):
        # type: () -> Iterator[pdart.pds4.Product.Product]
        """
        Generate the products of this :class:`~pdart.pds4.Collection`
        as :class:`~pdart.pds4.Product` objects.
        """
        from pdart.pds4.Product import Product

        dir_fp = self.absolute_filepath()
        for (dirpath, dirnames, filenames) in os.walk(dir_fp):
            for filename in filenames:
                (root, ext) = os.path.splitext(filename)
                if ext in Product.FILE_EXTS:
                    product_lid = LID('%s:%s' % (self.lid.lid, root))
                    yield Product(self.archive, product_lid)

    def bundle(self):
        # type: () -> pdart.pds4.Bundle.Bundle
        """Return the bundle this collection belongs to."""
        from pdart.pds4.Bundle import Bundle
        return Bundle(self.archive, self.lid.parent_lid())

    def browse_collection(self):
        # type: () -> Collection
        """Return the browse collection object for this collection."""
        return Collection(self.archive, self.lid.to_browse_lid())

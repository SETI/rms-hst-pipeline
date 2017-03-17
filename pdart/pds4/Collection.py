"""Representation of a PDS4 collection."""
import os.path
import re

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.

from pdart.pds4.Component import *
from pdart.pds4.LID import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator
    import pdart.pds4.Bundle
    import pdart.pds4.Product


class Collection(Component):
    """A PDS4 Collection."""

    DIRECTORY_PATTERN = r'\A(([a-z]+)_([a-z0-9]+)_([a-z0-9_]+)|document)\Z'
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
                        self.lid.collection_id).group(3)

    def prefix(self):
        # type: () -> unicode
        """
        Return the prefix for FITS files in this
        :class:`~pdart.pds4.Collection`.  It is calculated from the
        collection's :class:`~pdart.pds4.LID`.

        'data' and 'browse' are some possible results.
        """
        match = re.match(Collection.DIRECTORY_PATTERN,
                         self.lid.collection_id)
        # hack for document collections, which have no prefix or
        # suffix
        if match:
            return match.group(2)
        else:
            return None

    def suffix(self):
        # type: () -> unicode
        """
        Return the suffix for FITS files in this
        :class:`~pdart.pds4.Collection`.  It is calculated from the
        collection's :class:`~pdart.pds4.LID`.
        """
        match = re.match(Collection.DIRECTORY_PATTERN,
                         self.lid.collection_id)
        # hack for document collections, which have no prefix or
        # suffix
        if match:
            return match.group(4)
        else:
            return None

    def absolute_filepath(self):
        # type: () -> str
        """Return the absolute filepath to the collection's directory."""
        return os.path.join(self.archive.root,
                            self.lid.bundle_id, self.lid.collection_id)

    def label_filepath(self):
        # type: () -> str
        """Return the absolute filepath to the collection's label."""
        collection_fp = self.absolute_filepath()

        # hack for document collections, which have no prefix or
        # suffix
        pref = self.prefix()
        if pref:
            name = 'collection_%s.xml' % self.prefix()
        else:
            name = 'collection.xml'
        return os.path.join(collection_fp, name)

    def inventory_name(self):
        # type: () -> unicode
        """Return the filename of the collection's inventory."""

        # hack for document collections, which have no prefix or
        # suffix
        pref = self.prefix()
        if pref:
            return 'collection_%s.csv' % self.prefix()
        else:
            return 'collections.csv'

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
        if self.lid.collection_id == 'document':
            for subdir in os.listdir(dir_fp):
                product_lid = LID('%s:%s' % (self.lid.lid, subdir))
                yield Product(self.archive, product_lid)
        else:
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

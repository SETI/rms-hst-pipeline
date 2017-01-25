"""Representation of a PDS4 bundle."""
import os
import os.path
import re

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.

from pdart.pds4.Collection import Collection
from pdart.pds4.Component import *
from pdart.pds4.LID import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator
    import pdart.pds4.Archive
    import pdart.pds4.Product


class Bundle(Component):
    """A PDS4 Bundle."""

    DIRECTORY_PATTERN = r'\Ahst_([0-9]{5})\Z'
    # type: str
    """
    A regexp pattern for bundle directory names, used to validate
    directory names or to extract proposal ids.
    """

    def __init__(self, arch, lid):
        # type: (pdart.pds4.Archive.Archive, LID) -> None
        """
        Create a :class:`~pdart.pds4.Bundle.Bundle` given the
        :class:`~pdart.pds4.Archive.Archive` it lives in and its
        :class:`~pdart.pds4.LID.LID`.
        """
        assert lid.is_bundle_lid()
        super(Bundle, self).__init__(arch, lid)

    def __repr__(self):
        return 'Bundle(%r, %r)' % (self.archive, self.lid)

    def proposal_id(self):
        # type: () -> int
        """
        Return the proposal ID for this
        :class:`~pdart.pds4.Bundle.Bundle`.  It is calculated from the
        bundle's :class:`~pdart.pds4.LID.LID`.
        """
        return int(re.match(Bundle.DIRECTORY_PATTERN,
                            self.lid.bundle_id).group(1))

    def absolute_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the bundle's directory."""
        return os.path.join(self.archive.root, self.lid.bundle_id)

    def label_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the bundle's label."""
        return os.path.join(self.absolute_filepath(), 'bundle.xml')

    def collections(self):
        # type: () -> Iterator[Collection]
        """
        Generate the collections of this
        :class:`~pdart.pds4.Bundle.Bundle` as
        :class:`~pdart.pds4.Collection.Collection` objects.
        """
        dir_fp = self.absolute_filepath()
        for subdir in os.listdir(dir_fp):
            if re.match(Collection.DIRECTORY_PATTERN,
                        subdir):
                collection_lid = LID('%s:%s' % (self.lid.lid, subdir))
                yield Collection(self.archive, collection_lid)

    def products(self):
        # type: () -> Iterator[pdart.pds4.Product.Product]
        """
        Generate the products of this
        :class:`~pdart.pds4.Bundle.Bundle` as
        :class:`~pdart.pds4.Product.Product` objects.
        """
        for collection in self.collections():
            for product in collection.products():
                yield product

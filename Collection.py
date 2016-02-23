import os.path
import re

import ArchiveComponent
import Bundle
import LID
import Product


class Collection(ArchiveComponent.ArchiveComponent):
    """A PDS4 Collection."""

    DIRECTORY_PATTERN = '\Adata_([a-z0-9]+)_([a-z0-9_]+)\Z'

    def __init__(self, arch, lid):
        """
        Create a Collection given the archive it lives in and its LID.
        """
        assert lid.isCollectionLID()
        ArchiveComponent.ArchiveComponent.__init__(self, arch, lid)

    def __repr__(self):
        return 'Collection(%s, %s)' % (repr(self.archive), repr(self.lid))

    def directoryFilepath(self):
        """Return the absolute filepath to the component's directory."""
        return os.path.join(self.archive.root,
                            self.lid.bundleId, self.lid.collectionId)

    def products(self):
        """
        Generate the products of this bundle as Product objects.
        """
        dirFP = self.directoryFilepath()
        for subdir in os.listdir(dirFP):
            if re.match(Product.Product.DIRECTORY_PATTERN, subdir):
                productLID = LID.LID('%s:%s' % (self.lid.LID, subdir))
                yield Product.Product(self.archive, productLID)

    def instrument(self):
        """
        Return the instrument for this collection.  It is calculated
        from the collection's LID.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collectionId).group(1)

    def suffix(self):
        """
        Return the suffix for FITS files in this collection.  It is
        calculated from the collection's LID.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collectionId).group(2)

    def bundle(self):
        """Return the bundle this collection belongs to."""
        return Bundle.Bundle(self.archive, self.lid.parentLID())

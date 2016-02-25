import os.path
import re

import ArchiveComponent
import Bundle
import Collection


class Product(ArchiveComponent.ArchiveComponent):
    """A PDS4 Product."""

    DIRECTORY_PATTERN = '\Avisit_([a-z0-9]{2})\Z'

    def __init__(self, arch, lid):
        """
        Create a Product given the archive it lives in and its LID.
        """
        assert lid.isProductLID()
        super(Product, self).__init__(arch, lid)

    def __repr__(self):
        return 'Product(%r, %r)' % (self.archive, self.lid)

    def directoryFilepath(self):
        """Return the absolute filepath to the component's directory."""
        return os.path.join(self.archive.root, self.lid.bundleId,
                            self.lid.collectionId, self.lid.productId)

    def visit(self):
        """
        Return the visit code for this product.  It is calculated from
        the product's LID.
        """
        return re.match(Product.DIRECTORY_PATTERN,
                        self.lid.productId).group(1)

    def collection(self):
        """Return the collection this product belongs to."""
        return Collection.Collection(self.archive, self.lid.parentLID())

    def bundle(self):
        """Return the bundle this product belongs to."""
        return Bundle.Bundle(self.archive, self.lid.parentLID().parentLID())

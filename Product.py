import os.path
import re

import ArchiveComponent


class Product(ArchiveComponent.ArchiveComponent):
    DIRECTORY_PATTERN = '\Avisit_([a-z0-9]{2})\Z'

    def __init__(self, arch, lid):
        assert lid.isProductLID()
        ArchiveComponent.ArchiveComponent.__init__(self, arch, lid)

    def __repr__(self):
        return 'Product(%s, %s)' % (repr(self.archive), repr(self.lid))

    def directoryFilepath(self):
        return os.path.join(self.archive.root, self.lid.bundleId,
                            self.lid.collectionId, self.lid.productId)

    def visit(self):
        return re.match(Product.DIRECTORY_PATTERN,
                        self.lid.productId).group(1)

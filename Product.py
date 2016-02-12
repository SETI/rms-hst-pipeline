import os.path

import ArchiveComponent

class Product(ArchiveComponent.ArchiveComponent):
    DIRECTORY_PATTERN = '\Avisit_([a-z0-9]{2})\Z'

    def __init__(self, arch, lid):
        ArchiveComponent.ArchiveComponent.__init__(self, arch, lid)

    def __repr__(self):
        return 'Product(%s, %s)' % (repr(self.archive), repr(self.lid))
                                                   
    def directoryFilepath(self):
        return os.path.join(self.archive.root, self.lid.bundleID, 
                            self.lid.collectionID, self.lid.productID)


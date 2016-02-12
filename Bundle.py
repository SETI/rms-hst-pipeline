import os
import re

import ArchiveComponent
import Collection
import LID

class Bundle(ArchiveComponent.ArchiveComponent):
    DIRECTORY_PATTERN = '\Ahst_([0-9]{5})\Z'

    def __init__(self, arch, lid):
        assert lid.isBundleLID()
        ArchiveComponent.ArchiveComponent.__init__(self, arch, lid)

    def __repr__(self):
        return 'Bundle(%s, %s)' % (repr(self.archive), repr(self.lid))

    def directoryFilepath(self):
        return os.path.join(self.archive.root, self.lid.bundleId)

    def collections(self):
        dirFP = self.directoryFilepath()
        for subdir in os.listdir(dirFP):
            if re.match(Collection.Collection.DIRECTORY_PATTERN, subdir):
                collectionLID = LID.LID('%s:%s' % (self.lid.LID, subdir))
                yield Collection.Collection(self.archive, collectionLID)

    def products(self):
        for collection in self.collections():
            for product in collection.products():
                yield product


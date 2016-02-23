import os
import re

import ArchiveComponent
import Collection
import LID


class Bundle(ArchiveComponent.ArchiveComponent):
    """A PDS4 Bundle."""

    DIRECTORY_PATTERN = '\Ahst_([0-9]{5})\Z'

    def __init__(self, arch, lid):
        """
        Create a Bundle given the archive it lives in and its LID.
        """
        assert lid.isBundleLID()
        ArchiveComponent.ArchiveComponent.__init__(self, arch, lid)

    def __repr__(self):
        return 'Bundle(%s, %s)' % (repr(self.archive), repr(self.lid))

    def directoryFilepath(self):
        """Return the absolute filepath to the component's directory."""
        return os.path.join(self.archive.root, self.lid.bundleId)

    def collections(self):
        """
        Generate the collections of this bundle as Collection objects.
        """
        dirFP = self.directoryFilepath()
        for subdir in os.listdir(dirFP):
            if re.match(Collection.Collection.DIRECTORY_PATTERN, subdir):
                collectionLID = LID.LID('%s:%s' % (self.lid.LID, subdir))
                yield Collection.Collection(self.archive, collectionLID)

    def products(self):
        """Generate the products of this bundle as Product objects."""
        for collection in self.collections():
            for product in collection.products():
                yield product

    def proposalId(self):
        """
        Return the proposal ID for this bundle.  It is calculated
        from the bundle's LID.
        """
        return int(re.match(Bundle.DIRECTORY_PATTERN,
                            self.lid.bundleId).group(1))

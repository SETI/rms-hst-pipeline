import os
import re

import pdart.pds4.Component
import pdart.pds4.Collection
import pdart.pds4.LID


class Bundle(pdart.pds4.Component.Component):
    """A PDS4 Bundle."""

    DIRECTORY_PATTERN = r'\Ahst_([0-9]{5})\Z'

    def __init__(self, arch, lid):
        """
        Create a :class:`Bundle` given the :class:`Archive` it lives
        in and its :class:`LID`.
        """
        assert lid.is_bundle_lid()
        super(Bundle, self).__init__(arch, lid)

    def __repr__(self):
        return 'Bundle(%r, %r)' % (self.archive, self.lid)

    def absolute_filepath(self):
        """Return the absolute filepath to the component's directory."""
        return os.path.join(self.archive.root, self.lid.bundle_id)

    def collections(self):
        """
        Generate the collections of this :class:`Bundle` as
        :class:`Collection` objects.
        """
        dir_fp = self.absolute_filepath()
        for subdir in os.listdir(dir_fp):
            if re.match(pdart.pds4.Collection.Collection.DIRECTORY_PATTERN,
                        subdir):
                collection_lid = pdart.pds4.LID.LID('%s:%s' %
                                                    (self.lid.lid, subdir))
                yield pdart.pds4.Collection.Collection(self.archive,
                                                       collection_lid)

    def products(self):
        """
        Generate the products of this :class:`Bundle` as
        :class:`Product` objects.
        """
        for collection in self.collections():
            for product in collection.products():
                yield product

    def proposal_id(self):
        """
        Return the proposal ID for this :class:`Bundle`.  It is
        calculated from the bundle's :class:`LID`.
        """
        return int(re.match(Bundle.DIRECTORY_PATTERN,
                            self.lid.bundle_id).group(1))
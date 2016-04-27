import os.path
import re

import pdart.pds4.Bundle
import pdart.pds4.Component
import pdart.pds4.LID
import Product


class Collection(pdart.pds4.Component.Component):
    """A PDS4 Collection."""

    DIRECTORY_PATTERN = r'\Adata_([a-z0-9]+)_([a-z0-9_]+)\Z'

    def __init__(self, arch, lid):
        """
        Create a :class:`Collection` given the :class:`Archive` it
        lives in and its :class:`LID`.
        """
        assert lid.is_collection_lid()
        super(Collection, self).__init__(arch, lid)

    def __repr__(self):
        return 'Collection(%r, %r)' % (self.archive, self.lid)

    def absolute_filepath(self):
        """Return the absolute filepath to the component's directory."""
        return os.path.join(self.archive.root,
                            self.lid.bundle_id, self.lid.collection_id)

    def products(self):
        """
        Generate the products of this :class:`Collection` as
        :class:`Product` objects.
        """
        dir_fp = self.absolute_filepath()
        for (dirpath, dirnames, filenames) in os.walk(dir_fp):
            for filename in filenames:
                (root, ext) = os.path.splitext(filename)
                if ext == '.fits':
                    product_lid = pdart.pds4.LID.LID('%s:%s' %
                                                     (self.lid.lid, root))
                    yield Product.Product(self.archive, product_lid)

    def instrument(self):
        """
        Return the instrument for this :class:`Collection`.  It is
        calculated from the collection's :class:`LID`.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(1)

    def suffix(self):
        """
        Return the suffix for FITS files in this :class:`Collection`.
        It is calculated from the collection's :class:`LID`.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(2)

    def bundle(self):
        """Return the bundle this collection belongs to."""
        return pdart.pds4.Bundle.Bundle(self.archive, self.lid.parent_lid())
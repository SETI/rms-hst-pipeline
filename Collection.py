import os.path
import re

import ArchiveComponent
import Bundle
import LID
import Product


class Collection(ArchiveComponent.ArchiveComponent):
    """A PDS4 Collection."""

    DIRECTORY_PATTERN = r'\Adata_([a-z0-9]+)_([a-z0-9_]+)\Z'

    def __init__(self, arch, lid):
        """
        Create a Collection given the archive it lives in and its LID.
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
        Generate the products of this bundle as Product objects.
        """
        dir_fp = self.absolute_filepath()
        for (dirpath, dirnames, filenames) in os.walk(dir_fp):
            for filename in filenames:
                (root, ext) = os.path.splitext(filename)
                if ext == '.fits':
                    product_lid = LID.LID('%s:%s' % (self.lid.lid, root))
                    yield Product.Product(self.archive, product_lid)

    def instrument(self):
        """
        Return the instrument for this collection.  It is calculated
        from the collection's LID.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(1)

    def suffix(self):
        """
        Return the suffix for FITS files in this collection.  It is
        calculated from the collection's LID.
        """
        return re.match(Collection.DIRECTORY_PATTERN,
                        self.lid.collection_id).group(2)

    def bundle(self):
        """Return the bundle this collection belongs to."""
        return Bundle.Bundle(self.archive, self.lid.parent_lid())

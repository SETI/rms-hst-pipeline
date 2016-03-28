import os.path
import re
import shutil
import tempfile
import unittest

import ArchiveComponent
import ArchiveFile
import Bundle
import Collection
import FileArchive
import HstFilename
import LID


def _find_product_file(dir, filename):
    """
    Find a file by name in a directory or one of its subdirectories
    and return the absolute filepath.  Assume the directory path is
    absolute and that only one file with that name exists under the
    directory.
    """
    hstFilename = HstFilename.HstFilename(filename)
    visit = hstFilename.visit()
    return os.path.join(dir, 'visit_%s' % visit, filename)


class Product(ArchiveComponent.ArchiveComponent):
    """A PDS4 Product."""

    VISIT_DIRECTORY_PATTERN = r'\Avisit_([a-z0-9]{2})\Z'

    def __init__(self, arch, lid):
        """
        Create a Product given the archive it lives in and its LID.
        """
        assert lid.is_product_lid()
        super(Product, self).__init__(arch, lid)

    def __repr__(self):
        return 'Product(%r, %r)' % (self.archive, self.lid)

    def absolute_filepath(self):
        """Return the absolute filepath to the product file."""
        collection_fp = self.collection().absolute_filepath()
        res = _find_product_file(collection_fp,
                                       self.lid.product_id + '.fits')
        assert res, ('Product.absolute_filepath(%r) = %r '
                     'where collection_fp = %r' % (self, res, collection_fp))
        return res

    def visit(self):
        """
        Return the visit code for this product.  It is calculated from
        the product's filepath
        """
        fp = self.absolute_filepath()
        visit_fp = os.path.basename(os.path.dirname(fp))
        return re.match(Product.VISIT_DIRECTORY_PATTERN, visit_fp).group(1)

    def files(self):
        basename = os.path.basename(self.absolute_filepath())
        return [ArchiveFile.ArchiveFile(self, basename)]

    def absolute_filepath_is_directory(self):
        return False

    def collection(self):
        """Return the collection this product belongs to."""
        return Collection.Collection(self.archive, self.lid.parent_lid())

    def bundle(self):
        """Return the bundle this product belongs to."""
        return Bundle.Bundle(self.archive, self.lid.parent_lid().parent_lid())


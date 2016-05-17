import os.path
import re
import shutil
import tempfile
import unittest

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.

from pdart.pds4.Component import *
from pdart.pds4.File import *
from pdart.pds4.HstFilename import *


def _find_product_file(visit_dir, product_id):
    """
    Find a file by name in a directory or one of its subdirectories
    and return the absolute filepath.  Assume the directory path is
    absolute and that only one file with that name exists under the
    directory.  Return None on failure.
    """
    for ext in Product.FILE_EXTS:
        filepath = os.path.join(visit_dir, product_id + ext)
        if os.path.isfile(filepath):
            return filepath
    return None


class Product(Component):
    """A PDS4 Product."""

    VISIT_DIRECTORY_PATTERN = r'\Avisit_([a-z0-9]{2})\Z'
    FILE_EXTS = ['.fits', '.jpg']

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
        visit_fp = self.visit_filepath()
        res = _find_product_file(visit_fp, self.lid.product_id)

        collection_fp = self.collection().absolute_filepath()
        assert res, ('Product.absolute_filepath(%r) = %r '
                     'where collection_fp = %r' % (self, res, collection_fp))
        return res

    def visit_filepath(self):
        hst_filename = HstFilename(self.lid.product_id)
        collection_filepath = self.collection().absolute_filepath()
        visit_segment = 'visit_%s' % self.visit()
        return os.path.join(collection_filepath, visit_segment)

    def visit(self):
        """
        Return the visit code for this product.  It is calculated from
        the product's filepath
        """
        hst_filename = HstFilename(self.lid.product_id)
        return hst_filename.visit()

    def files(self):
        basename = os.path.basename(self.absolute_filepath())
        yield File(self, basename)

    def absolute_filepath_is_directory(self):
        return False

    def collection(self):
        """Return the collection this product belongs to."""
        from pdart.pds4.Collection import Collection
        return Collection(self.archive,
                          self.lid.parent_lid())

    def bundle(self):
        """Return the bundle this product belongs to."""
        from pdart.pds4.Bundle import Bundle
        return Bundle(self.archive,
                      self.lid.parent_lid().parent_lid())

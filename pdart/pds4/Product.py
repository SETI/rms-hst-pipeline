"""Representation of a PDS4 product."""
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

from typing import Iterator, TYPE_CHECKING
if TYPE_CHECKING:
    import pdart.pds4.Archive
    import pdart.pds4.Bundle
    import pdart.pds4.Collection
    import pdart.pds4.LID


def _find_product_file(visit_dir, product_id):
    # type: (unicode, unicode) -> unicode
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
    # type: str
    """
    A regexp pattern for product visit directory names, used to
    validate visit directory names or to extract visit ids.
    """

    FILE_EXTS = ['.fits', '.jpg']
    # type: List[unicode]
    """Currently legal file extensions for product files."""

    def __init__(self, arch, lid):
        # type: (pdart.pds4.Archive.Archive, pdart.pds4.LID.LID) -> None
        """
        Create a Product given the archive it lives in and its LID.
        """
        assert lid.is_product_lid()
        super(Product, self).__init__(arch, lid)

    def __repr__(self):
        return 'Product(%r, %r)' % (self.archive, self.lid)

    def absolute_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the product file."""
        visit_fp = self.visit_filepath()
        res = _find_product_file(visit_fp, self.lid.product_id)

        collection_fp = self.collection().absolute_filepath()
        assert res, ('Couldn\'t find any product files: '
                     'Product.absolute_filepath(%r) = %r '
                     'where collection_fp = %r' % (self, res, collection_fp))
        return res

    def label_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the product's label."""
        product_fp = self.absolute_filepath()
        (dir, product_basename) = os.path.split(product_fp)
        (root, ext) = os.path.splitext(product_basename)
        label_basename = root + '.xml'
        return os.path.join(dir, label_basename)

    def visit_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the product's visit directory."""
        hst_filename = HstFilename(self.lid.product_id)
        collection_filepath = self.collection().absolute_filepath()
        visit_segment = 'visit_%s' % self.visit()
        return os.path.join(collection_filepath, visit_segment)

    def visit(self):
        # type: () -> unicode
        """
        Return the visit id for this product.  It is calculated from
        the product's filepath.
        """
        hst_filename = HstFilename(self.lid.product_id)
        return hst_filename.visit()

    def files(self):
        # type: () -> Iterator[File]
        """
        Generate all the files belonging to this
        :class:`~pdart.pds4.Product.Product` as
        :class:`~pdart.pds4.File.File` objects.
        """
        basename = os.path.basename(self.absolute_filepath())
        yield File(self, basename)

    def absolute_filepath_is_directory(self):
        # type: () -> bool
        """
        Return True iff the product's absolute filepath is a
        directory.

        Always False because products' filepaths are to their
        (currently single) file.
        """
        return False

    def collection(self):
        # type: () -> pdart.pds4.Collection.Collection
        """Return the collection this product belongs to."""
        from pdart.pds4.Collection import Collection
        return Collection(self.archive,
                          self.lid.parent_lid())

    def bundle(self):
        # type: () -> pdart.pds4.Bundle.Bundle
        """Return the bundle this product belongs to."""
        from pdart.pds4.Bundle import Bundle
        return Bundle(self.archive,
                      self.lid.parent_lid().parent_lid())

    def browse_product(self):
        # type: () -> Product
        """Return the browse product object for this product."""
        return Product(self.archive, self.lid.to_browse_lid())

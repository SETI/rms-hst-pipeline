"""Representation of a PDS4 product."""
import re
import shutil
import tempfile
import unittest

from fs.path import basename, join, split, splitext

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.

from pdart.pds4.Component import *
from pdart.pds4.File import *
from pdart.pds4.HstFilename import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator
    from fs.base import FS
    import pdart.pds4.Archive
    import pdart.pds4.Bundle
    import pdart.pds4.Collection
    import pdart.pds4.LID


def _find_product_file(fs, visit_dir, product_id):
    # type: (FS, unicode, unicode) -> unicode
    """
    Find a file by name in a directory or one of its subdirectories
    and return the relative filepath.  Assume the directory path is
    relative and that only one file with that name exists under the
    directory.  Return None on failure.
    """
    for ext in Product.FILE_EXTS:
        filepath = join(visit_dir, product_id + ext)
        if fs.isfile(filepath):
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

    DATA_EXTS = ['.fits']
    # type: List[unicode]
    """Currently legal file extensions for data product files."""

    BROWSE_EXTS = ['.jpg']
    # type: List[unicode]
    """Currently legal file extensions for browse product files."""

    DOC_EXTS = ['.apt', '.pdf', '.pro', '.prop']
    # type: List[unicode]
    """Currently legal file extensions for document product files."""

    FILE_EXTS = DATA_EXTS + BROWSE_EXTS + DOC_EXTS
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

    def is_document_product(self):
        return self.lid.collection_id == 'document'

    def absolute_filepath(self):
        # type: () -> unicode
        """
        Return the absolute filepath to the product directory, if this
        is a document product, otherwise, to the product file.
        """
        if self.is_document_product():
            collection_filepath = self.collection().absolute_filepath()
            return join(collection_filepath, self.lid.product_id)
        else:
            visit_fp = self.relative_visit_filepath()
            root_fs = self.archive.root_fs
            rel = _find_product_file(root_fs, visit_fp, self.lid.product_id)
            res = root_fs.getsyspath(rel)

            collection_fp = self.collection().absolute_filepath()
            assert res, ('Couldn\'t find any product files: '
                         'Product.absolute_filepath(%r) = %r '
                         'where collection_fp = %r' % (self,
                                                       res,
                                                       collection_fp))
            return res

    def relative_filepath(self):
        # type: () -> unicode
        """
        Return the relative filepath to the product directory, if this
        is a document product, otherwise, to the product file.
        """
        if self.is_document_product():
            collection_filepath = self.collection().relative_filepath()
            return join(collection_filepath, self.lid.product_id)
        else:
            visit_fp = self.relative_visit_filepath()
            res = _find_product_file(self.archive.root_fs,
                                     visit_fp,
                                     self.lid.product_id)

            collection_fp = self.collection().relative_filepath()
            assert res, ('Couldn\'t find any product files: '
                         'Product.relative_filepath(%r) = %r '
                         'where collection_fp = %r' % (self,
                                                       res,
                                                       collection_fp))
            return res

    def label_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the product's label."""
        if self.is_document_product():
            # TODO should it be document.xml or phase2.xml or what?
            return join(self.absolute_filepath(), 'document.xml')
        else:
            product_fp = self.absolute_filepath()
            (dir, product_basename) = split(product_fp)
            (root, ext) = splitext(product_basename)
            label_basename = root + '.xml'
        return join(dir, label_basename)

    def visit_filepath(self):
        # type: () -> unicode
        """Return the absolute filepath to the product's visit directory."""
        assert not self.is_document_product()
        collection_filepath = self.collection().absolute_filepath()
        visit_segment = 'visit_%s' % self.visit()
        return join(collection_filepath, visit_segment)

    def relative_visit_filepath(self):
        # type: () -> unicode
        """Return the relative filepath to the product's visit directory."""
        assert not self.is_document_product()
        collection_filepath = self.collection().relative_filepath()
        visit_segment = 'visit_%s' % self.visit()
        return join(collection_filepath, visit_segment)

    def visit(self):
        # type: () -> unicode
        """
        Return the visit id for this product.  It is calculated from
        the product's filepath.
        """
        assert not self.is_document_product()
        hst_filename = HstFilename(self.lid.product_id)
        return hst_filename.visit()

    def files(self):
        # type: () -> Iterator[File]
        """
        Generate all the files belonging to this
        :class:`~pdart.pds4.Product.Product` as
        :class:`~pdart.pds4.File.File` objects.
        """
        if self.is_document_product():
            root_fs = self.archive.root_fs
            for filename in root_fs.listdir(self.relative_filepath()):
                if splitext(filename)[1] in Product.DOC_EXTS:
                    yield File(self, filename)
        else:
            filename = basename(self.relative_filepath())
            yield File(self, filename)

    def absolute_filepath_is_directory(self):
        # type: () -> bool
        """
        Return True iff the product's absolute filepath is a
        directory.

        Always False because products' filepaths are to their
        (currently single) file.
        """
        return self.is_document_product()

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
        assert not self.is_document_product()
        return Product(self.archive, self.lid.to_browse_lid())

    def first_filepath(self):
        # type: () -> unicode
        """Return the full filepath of the first file for this product."""
        return self.files().next().full_filepath()

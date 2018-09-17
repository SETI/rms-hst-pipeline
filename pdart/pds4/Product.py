"""Representation of a PDS4 product."""

from fs.path import basename, join, split, splitext
from typing import TYPE_CHECKING

from pdart.pds4.Component import Component
from pdart.pds4.File import File
from pdart.pds4.HstFilename import HstFilename

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.
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

"""Representation of a PDS4 bundle."""
import re
from typing import TYPE_CHECKING

from fs.path import join

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.

from pdart.pds4.Collection import Collection
from pdart.pds4.Component import Component
from pdart.pds4.LID import LID

if TYPE_CHECKING:
    from typing import Iterator
    import pdart.pds4.Product


class Bundle(Component):
    """A PDS4 Bundle."""

    DIRECTORY_PATTERN = r'\Ahst_([0-9]{5})\Z'  # type: str
    """
    A regexp pattern for bundle directory names, used to validate
    directory names or to extract proposal ids.
    """

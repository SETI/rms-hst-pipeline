"""Representation of a PDS4 collection."""
import re

from fs.path import join, splitext
from typing import TYPE_CHECKING

from pdart.pds4.Component import Component
from pdart.pds4.LID import LID

# We only import PDS4 subcomponent modules to avoid circular imports.
# If you want to import a supercomponent module, do it within a
# function or method.
if TYPE_CHECKING:
    from typing import Iterator
    import pdart.pds4.Bundle
    import pdart.pds4.Product


class Collection(Component):
    """A PDS4 Collection."""

    DIRECTORY_PATTERN \
        = r'\A(([a-z]+)_([a-z0-9]+)_([a-z0-9_]+)|document)\Z'  # type: str

import os.path
import re

from pdart.pds4.Bundle import *
from pdart.pds4.LID import *


class Archive(object):
    """An :class:`Archive` containing PDS4 Bundles."""

    def __init__(self, root):
        """
        Create an :class:`Archive` given a filepath to an existing
        directory.
        """
        assert os.path.exists(root) and os.path.isdir(root)
        self.root = root

    def __eq__(self, other):
        return self.root == other.root

    def __str__(self):
        return repr(self.root)

    def __repr__(self):
        return 'Archive(%r)' % self.root

    # Verifying parts

    @staticmethod
    def is_valid_instrument(inst):
        """Return True iff the argument is a valid instrument name."""
        return inst in ['acs', 'wfc3', 'wfpc2']

    @staticmethod
    def is_valid_proposal(prop):
        """Return True iff the argument is a valid integer HST proposal ID."""
        return isinstance(prop, int) and 0 <= prop and prop <= 99999

    @staticmethod
    def is_valid_visit(vis):
        """Return True iff the argument is a valid visit ID."""
        try:
            return re.match(r'\A[a-z0-9][a-z0-9]\Z', vis) is not None
        except:
            return False

    @staticmethod
    def is_valid_bundle_dir_basename(dirname):
        return re.match(r'\Ahst_[0-9]{5}\Z', dirname) is not None

    @staticmethod
    def is_valid_collection_dir_basename(dirname):
        return re.match(r'\Adata_[a-z0-9]+_', dirname) is not None

    @staticmethod
    def is_valid_product_dir_basename(dirname):
        return re.match(r'\Avisit_[a-z0-9]{2}\Z', dirname) is not None

    @staticmethod
    def is_hidden_file_basename(basename):
        return basename[0] == '.'

    # Walking the hierarchy with objects
    def bundles(self):
        """Generate the bundles stored in this :class:`Archive`."""
        for subdir in os.listdir(self.root):
            if re.match(Bundle.DIRECTORY_PATTERN, subdir):
                bundle_lid = LID('urn:nasa:pds:%s' % subdir)
                yield Bundle(self, bundle_lid)

    def collections(self):
        """Generate the collections stored in this :class:`Archive`."""
        for b in self.bundles():
            for c in b.collections():
                yield c

    def products(self):
        """Generate the products stored in this :class:`Archive`."""
        for b in self.bundles():
            for c in b.collections():
                for p in c.products():
                    yield p
